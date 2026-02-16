import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_PATH = "src/data/epl_final.csv"
TEST_SEASON = "2024/25"
ROLLING_WINDOW = 5
ELO_BASE = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 60.0
ELO_K_GRID = [10.0, 20.0, 30.0]
ELO_HOME_ADV_GRID = [20.0, 60.0, 100.0]

# Use a cheaper model during Elo tuning to reduce runtime.
TUNE_N_ESTIMATORS = 120
TUNE_MAX_DEPTH = 4

# Use a stronger model for final training/evaluation.
FINAL_N_ESTIMATORS = 400
FINAL_MAX_DEPTH = 6

# Columns known only during/after the match.
LEAKY_MATCH_EVENT_COLUMNS = [
    "FullTimeHomeGoals",
    "FullTimeAwayGoals",
    "HalfTimeHomeGoals",
    "HalfTimeAwayGoals",
    "HalfTimeResult",
    "HomeShots",
    "AwayShots",
    "HomeShotsOnTarget",
    "AwayShotsOnTarget",
    "HomeCorners",
    "AwayCorners",
    "HomeFouls",
    "AwayFouls",
    "HomeYellowCards",
    "AwayYellowCards",
    "HomeRedCards",
    "AwayRedCards",
]


def make_team_form_features(df, window=5):
    work = df.copy()
    work["MatchDate"] = pd.to_datetime(work["MatchDate"])
    work["match_id"] = range(len(work))

    # Build one row per team per match.
    home_rows = pd.DataFrame(
        {
            "match_id": work["match_id"],
            "MatchDate": work["MatchDate"],
            "team": work["HomeTeam"],
            "side": "home",
            "goals_for": work["FullTimeHomeGoals"],
            "goals_against": work["FullTimeAwayGoals"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "match_id": work["match_id"],
            "MatchDate": work["MatchDate"],
            "team": work["AwayTeam"],
            "side": "away",
            "goals_for": work["FullTimeAwayGoals"],
            "goals_against": work["FullTimeHomeGoals"],
        }
    )

    team_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    team_rows["goal_diff"] = team_rows["goals_for"] - team_rows["goals_against"]

    # Points from each team's perspective.
    team_rows["points"] = 0
    home_win_mask = (team_rows["side"] == "home") & (team_rows["goal_diff"] > 0)
    away_win_mask = (team_rows["side"] == "away") & (team_rows["goal_diff"] > 0)
    draw_mask = team_rows["goal_diff"] == 0
    team_rows.loc[home_win_mask | away_win_mask, "points"] = 3
    team_rows.loc[draw_mask, "points"] = 1

    team_rows = team_rows.sort_values(["team", "MatchDate", "match_id"]).reset_index(drop=True)

    by_team = team_rows.groupby("team", group_keys=False)
    by_team_side = team_rows.groupby(["team", "side"], group_keys=False)

    # Historical form before current match.
    team_rows["matches_played_before"] = by_team.cumcount()
    team_rows["points_cum_before"] = by_team["points"].cumsum() - team_rows["points"]
    team_rows["goal_diff_cum_before"] = by_team["goal_diff"].cumsum() - team_rows["goal_diff"]
    team_rows["goals_for_cum_before"] = by_team["goals_for"].cumsum() - team_rows["goals_for"]
    team_rows["goals_against_cum_before"] = (
        by_team["goals_against"].cumsum() - team_rows["goals_against"]
    )

    team_rows["points_last5"] = by_team["points"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    team_rows["goal_diff_last5"] = by_team["goal_diff"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    team_rows["goals_for_last5"] = by_team["goals_for"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    team_rows["goals_against_last5"] = by_team["goals_against"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )

    team_rows["matches_side_before"] = by_team_side.cumcount()
    team_rows["points_side_last5"] = by_team_side["points"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    team_rows["goal_diff_side_last5"] = by_team_side["goal_diff"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )

    feature_cols = [
        "match_id",
        "matches_played_before",
        "points_cum_before",
        "goal_diff_cum_before",
        "goals_for_cum_before",
        "goals_against_cum_before",
        "points_last5",
        "goal_diff_last5",
        "goals_for_last5",
        "goals_against_last5",
        "matches_side_before",
        "points_side_last5",
        "goal_diff_side_last5",
    ]

    home_form = team_rows[team_rows["side"] == "home"][feature_cols].copy()
    away_form = team_rows[team_rows["side"] == "away"][feature_cols].copy()

    home_form = home_form.rename(
        columns={c: f"home_{c}" for c in feature_cols if c != "match_id"}
    )
    away_form = away_form.rename(
        columns={c: f"away_{c}" for c in feature_cols if c != "match_id"}
    )

    out = work.merge(home_form, on="match_id", how="left").merge(away_form, on="match_id", how="left")
    return out.drop(columns=["match_id"])


def add_elo_features(df, base_rating=1500.0, k_factor=20.0, home_advantage=60.0):
    work = df.copy()
    work["MatchDate"] = pd.to_datetime(work["MatchDate"])
    work["match_id"] = range(len(work))

    ordered = work.sort_values(["MatchDate", "match_id"]).copy()

    ratings = {}
    home_elo_pre = []
    away_elo_pre = []
    elo_diff_pre = []

    for row in ordered.itertuples(index=False):
        home = row.HomeTeam
        away = row.AwayTeam

        home_rating = ratings.get(home, base_rating)
        away_rating = ratings.get(away, base_rating)

        home_elo_pre.append(home_rating)
        away_elo_pre.append(away_rating)
        elo_diff_pre.append(home_rating - away_rating)

        # Expected score with home advantage.
        home_expected = 1.0 / (1.0 + 10 ** ((away_rating - (home_rating + home_advantage)) / 400.0))
        away_expected = 1.0 - home_expected

        if row.FullTimeHomeGoals > row.FullTimeAwayGoals:
            home_actual, away_actual = 1.0, 0.0
        elif row.FullTimeHomeGoals < row.FullTimeAwayGoals:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5

        ratings[home] = home_rating + k_factor * (home_actual - home_expected)
        ratings[away] = away_rating + k_factor * (away_actual - away_expected)

    ordered["home_elo_pre"] = home_elo_pre
    ordered["away_elo_pre"] = away_elo_pre
    ordered["elo_diff_pre"] = elo_diff_pre

    out = work.merge(
        ordered[["match_id", "home_elo_pre", "away_elo_pre", "elo_diff_pre"]],
        on="match_id",
        how="left",
    )
    return out.drop(columns=["match_id"])


def print_block(title):
    print(f"\n=== {title} ===")


def print_metrics(y_true, y_pred, y_prob):
    print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:          {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Log Loss:          {log_loss(y_true, y_prob):.4f}")


def build_pipeline(cat_cols, num_cols, n_estimators, max_depth):
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def prepare_dataset(df_with_form, k_factor, home_advantage):
    out = add_elo_features(
        df_with_form,
        base_rating=ELO_BASE,
        k_factor=k_factor,
        home_advantage=home_advantage,
    )
    label_map = {"H": 0, "D": 1, "A": 2}
    out["y"] = out["FullTimeResult"].map(label_map)
    out = out.drop(columns=["FullTimeResult"])

    # Fill early-season missing form values with zeros.
    form_cols = [c for c in out.columns if c.startswith("home_") or c.startswith("away_")]
    out[form_cols] = out[form_cols].fillna(0)
    return out


def split_xy(df, train_mask, val_mask, drop_feature_cols):
    train_part = df[train_mask].copy()
    val_part = df[val_mask].copy()
    if train_part.empty or val_part.empty:
        return None

    x_train = train_part.drop(columns=["y"] + drop_feature_cols)
    y_train_local = train_part["y"]
    x_val = val_part.drop(columns=["y"] + drop_feature_cols)
    y_val_local = val_part["y"]
    return x_train, y_train_local, x_val, y_val_local


def tune_elo_params(df_with_form, drop_feature_cols):
    train_seasons = sorted(s for s in df_with_form["Season"].unique() if s < TEST_SEASON)
    if len(train_seasons) < 2:
        return ELO_K, ELO_HOME_ADV, None, None

    tune_season = train_seasons[-1]
    best_score = float("inf")
    best_k = ELO_K
    best_home_adv = ELO_HOME_ADV

    for k in ELO_K_GRID:
        for home_adv in ELO_HOME_ADV_GRID:
            tmp = prepare_dataset(df_with_form, k_factor=k, home_advantage=home_adv)
            split = split_xy(
                tmp,
                train_mask=tmp["Season"] < tune_season,
                val_mask=tmp["Season"] == tune_season,
                drop_feature_cols=drop_feature_cols,
            )
            if split is None:
                continue

            x_train_local, y_train_local, x_val_local, y_val_local = split
            cat_cols_local = x_train_local.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            num_cols_local = x_train_local.select_dtypes(
                exclude=["object", "category"]
            ).columns.tolist()

            pipe = build_pipeline(
                cat_cols_local,
                num_cols_local,
                n_estimators=TUNE_N_ESTIMATORS,
                max_depth=TUNE_MAX_DEPTH,
            )
            pipe.fit(x_train_local, y_train_local)
            probs_local = pipe.predict_proba(x_val_local)
            score = log_loss(y_val_local, probs_local)

            if score < best_score:
                best_score = score
                best_k = k
                best_home_adv = home_adv

    return best_k, best_home_adv, tune_season, best_score


raw_df = pd.read_csv(DATA_PATH)
df_with_form = make_team_form_features(raw_df, window=ROLLING_WINDOW)

drop_feature_cols = ["Season", "MatchDate"] + LEAKY_MATCH_EVENT_COLUMNS
drop_feature_cols = [c for c in drop_feature_cols if c in df_with_form.columns]

best_k, best_home_adv, tune_season, tune_score = tune_elo_params(
    df_with_form, drop_feature_cols
)
df = prepare_dataset(df_with_form, k_factor=best_k, home_advantage=best_home_adv)

# Split before fitting transforms.
train = df[df["Season"] < TEST_SEASON].copy()
test = df[df["Season"] == TEST_SEASON].copy()

if train.empty or test.empty:
    raise ValueError(
        f"Train/Test split is empty. Check TEST_SEASON={TEST_SEASON} and Season values."
    )

drop_feature_cols = ["Season", "MatchDate"] + LEAKY_MATCH_EVENT_COLUMNS
drop_feature_cols = [c for c in drop_feature_cols if c in df.columns]

X_train = train.drop(columns=["y"] + drop_feature_cols)
y_train = train["y"]
X_val = test.drop(columns=["y"] + drop_feature_cols)
y_val = test["y"]

cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
clf = build_pipeline(
    cat_cols,
    num_cols,
    n_estimators=FINAL_N_ESTIMATORS,
    max_depth=FINAL_MAX_DEPTH,
)

clf.fit(X_train, y_train)
probs = clf.predict_proba(X_val)
preds = probs.argmax(axis=1)

majority_baseline = DummyClassifier(strategy="most_frequent")
majority_baseline.fit(X_train, y_train)
majority_preds = majority_baseline.predict(X_val)
majority_probs = majority_baseline.predict_proba(X_val)

prior_baseline = DummyClassifier(strategy="prior")
prior_baseline.fit(X_train, y_train)
prior_preds = prior_baseline.predict(X_val)
prior_probs = prior_baseline.predict_proba(X_val)

print("Train rows:", len(X_train), "Test rows:", len(X_val))
print("Feature count:", X_train.shape[1])
if tune_season is not None:
    print(
        f"Elo tuning season: {tune_season} | best K={best_k:.1f}, "
        f"home_adv={best_home_adv:.1f}, log_loss={tune_score:.4f}"
    )
else:
    print(f"Elo tuning skipped (insufficient seasons). Using K={best_k:.1f}, home_adv={best_home_adv:.1f}")

print_block("Baseline: Most Frequent")
print_metrics(y_val, majority_preds, majority_probs)

print_block("Baseline: Class Prior Probabilities")
print_metrics(y_val, prior_preds, prior_probs)

print_block("Model: XGBoost")
print_metrics(y_val, preds, probs)
print(f"Weighted F1:       {f1_score(y_val, preds, average='weighted'):.4f}")
print(f"ROC-AUC (OvR):     {roc_auc_score(y_val, probs, multi_class='ovr'):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_val, preds))
print(
    "Classification Report:\n",
    classification_report(y_val, preds, target_names=["H", "D", "A"]),
)
