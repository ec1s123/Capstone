from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Prem-2026-2003"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

TEST_SEASON = "2024/25"
TEST_SEASON_START = 2024
RANDOM_SEED = 42
CLASS_LABELS = np.array(["H", "D", "A"])
RESULT_NAMES = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
LABEL_MAP = {"H": 0, "D": 1, "A": 2}

BASE_NUMERIC_COLUMNS = ["month", "day_of_week"]
ODDS_NUMERIC_COLUMNS = BASE_NUMERIC_COLUMNS + [
    "prob_home",
    "prob_draw",
    "prob_away",
    "market_margin",
    "prob_gap",
    "favorite_prob",
    "log_b365h",
    "log_b365d",
    "log_b365a",
]

TEAM_NORMALISATION = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "QPR": "Queens Park Rangers",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
}

BASELINE_CONFIG = {"learning_rate": 0.02, "epochs": 1500, "l2": 1e-4, "batch_size": 512}
ODDS_MODEL_CONFIG = {"learning_rate": 0.01, "epochs": 2000, "l2": 5e-4, "batch_size": 512}


@dataclass
class MatrixSpec:
    home_levels: list[str]
    away_levels: list[str]
    numeric_columns: list[str]
    numeric_means: np.ndarray
    numeric_stds: np.ndarray


@dataclass
class SoftmaxModel:
    weights: np.ndarray
    bias: np.ndarray
    spec: MatrixSpec


def read_csv_robust(path: Path) -> pd.DataFrame:
    rows: list[list[str]] | None = None
    for encoding in ("utf-8", "latin1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as handle:
                rows = list(csv.reader(handle))
            break
        except UnicodeDecodeError:
            continue

    if not rows:
        raise ValueError(f"Could not read {path}")

    header, *body = rows
    width = len(header)
    cleaned_rows: list[list[str]] = []
    for row in body:
        if not row:
            continue
        if len(row) > width:
            row = row[:width]
        elif len(row) < width:
            row = row + [""] * (width - len(row))
        cleaned_rows.append(row)

    frame = pd.DataFrame(cleaned_rows, columns=header)
    frame.columns = [str(column).replace("\xa0", " ").strip() for column in frame.columns]
    object_columns = frame.select_dtypes(include="object").columns
    for column in object_columns:
        frame[column] = (
            frame[column]
            .astype(str)
            .str.replace("\xa0", " ", regex=False)
            .str.strip()
            .replace({"": pd.NA})
        )
    return frame


def season_start_year(match_date: pd.Timestamp) -> int:
    return match_date.year if match_date.month >= 7 else match_date.year - 1


def season_label(start_year: int) -> str:
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def load_match_data() -> pd.DataFrame:
    frames = [read_csv_robust(path) for path in sorted(DATA_DIR.glob("*.csv"))]
    matches = pd.concat(frames, ignore_index=True, sort=False).copy()

    matches["MatchDate"] = pd.to_datetime(
        matches["Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )
    matches = matches[matches["MatchDate"].notna()].copy()
    matches["SeasonStart"] = matches["MatchDate"].apply(season_start_year).astype(int)
    matches["Season"] = matches["SeasonStart"].apply(season_label)
    matches["HomeTeam"] = matches["HomeTeam"].replace(TEAM_NORMALISATION)
    matches["AwayTeam"] = matches["AwayTeam"].replace(TEAM_NORMALISATION)
    matches["y"] = matches["FTR"].map(LABEL_MAP)

    for column in ("B365H", "B365D", "B365A"):
        matches[column] = pd.to_numeric(matches[column], errors="coerce")

    matches = matches.dropna(
        subset=["HomeTeam", "AwayTeam", "MatchDate", "FTR", "y", "B365H", "B365D", "B365A"]
    ).copy()

    inverse_odds = 1.0 / matches[["B365H", "B365D", "B365A"]].to_numpy(dtype=np.float64)
    market_margin = inverse_odds.sum(axis=1, keepdims=True)
    implied_probabilities = inverse_odds / market_margin

    feature_frame = pd.DataFrame(
        {
            "month": matches["MatchDate"].dt.month.astype(np.float64).to_numpy(),
            "day_of_week": matches["MatchDate"].dt.dayofweek.astype(np.float64).to_numpy(),
            "prob_home": implied_probabilities[:, 0],
            "prob_draw": implied_probabilities[:, 1],
            "prob_away": implied_probabilities[:, 2],
            "market_margin": market_margin[:, 0],
            "prob_gap": implied_probabilities[:, 0] - implied_probabilities[:, 2],
            "favorite_prob": implied_probabilities.max(axis=1),
            "log_b365h": np.log(matches["B365H"].to_numpy(dtype=np.float64)),
            "log_b365d": np.log(matches["B365D"].to_numpy(dtype=np.float64)),
            "log_b365a": np.log(matches["B365A"].to_numpy(dtype=np.float64)),
        },
        index=matches.index,
    )

    matches = pd.concat([matches, feature_frame], axis=1)
    matches = matches.sort_values(["MatchDate", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return matches


def fit_matrix_spec(frame: pd.DataFrame, numeric_columns: list[str]) -> MatrixSpec:
    numeric_means = frame[numeric_columns].mean().to_numpy(dtype=np.float64)
    numeric_stds = frame[numeric_columns].std(ddof=0).replace(0, 1).to_numpy(dtype=np.float64)
    return MatrixSpec(
        home_levels=sorted(frame["HomeTeam"].unique().tolist()),
        away_levels=sorted(frame["AwayTeam"].unique().tolist()),
        numeric_columns=list(numeric_columns),
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
    )


def build_design_matrix(frame: pd.DataFrame, spec: MatrixSpec) -> np.ndarray:
    home_index = {team: index for index, team in enumerate(spec.home_levels)}
    away_index = {team: index for index, team in enumerate(spec.away_levels)}
    home_width = len(spec.home_levels)
    away_width = len(spec.away_levels)
    numeric_width = len(spec.numeric_columns)

    matrix = np.zeros((len(frame), home_width + away_width + numeric_width), dtype=np.float64)

    for row_index, team in enumerate(frame["HomeTeam"]):
        column_index = home_index.get(team)
        if column_index is not None:
            matrix[row_index, column_index] = 1.0

    away_offset = home_width
    for row_index, team in enumerate(frame["AwayTeam"]):
        column_index = away_index.get(team)
        if column_index is not None:
            matrix[row_index, away_offset + column_index] = 1.0

    numeric_values = frame[spec.numeric_columns].to_numpy(dtype=np.float64)
    numeric_values = (numeric_values - spec.numeric_means) / spec.numeric_stds
    numeric_values = np.nan_to_num(numeric_values, nan=0.0)
    matrix[:, away_offset + away_width :] = numeric_values
    return matrix


def fit_softmax_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    spec: MatrixSpec,
    config: dict[str, float | int],
) -> SoftmaxModel:
    rng = np.random.default_rng(RANDOM_SEED)
    sample_count, feature_count = features.shape
    class_count = len(CLASS_LABELS)

    weights = rng.normal(0, 0.01, size=(feature_count, class_count))
    bias = np.zeros(class_count, dtype=np.float64)

    first_moment_weights = np.zeros_like(weights)
    second_moment_weights = np.zeros_like(weights)
    first_moment_bias = np.zeros_like(bias)
    second_moment_bias = np.zeros_like(bias)

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    one_hot = np.eye(class_count)
    step = 0

    learning_rate = float(config["learning_rate"])
    epochs = int(config["epochs"])
    l2_penalty = float(config["l2"])
    batch_size = int(config["batch_size"])

    for _ in range(epochs):
        batch_order = rng.permutation(sample_count)
        for start_index in range(0, sample_count, batch_size):
            batch_index = batch_order[start_index : start_index + batch_size]
            batch_features = features[batch_index]
            batch_labels = labels[batch_index]

            logits = batch_features @ weights + bias
            logits -= logits.max(axis=1, keepdims=True)
            exponentiated = np.exp(logits)
            probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)

            gradient = (probabilities - one_hot[batch_labels]) / len(batch_index)
            gradient_weights = batch_features.T @ gradient + l2_penalty * weights
            gradient_bias = gradient.sum(axis=0)

            step += 1
            first_moment_weights = beta1 * first_moment_weights + (1 - beta1) * gradient_weights
            second_moment_weights = beta2 * second_moment_weights + (1 - beta2) * (
                gradient_weights**2
            )
            first_moment_bias = beta1 * first_moment_bias + (1 - beta1) * gradient_bias
            second_moment_bias = beta2 * second_moment_bias + (1 - beta2) * (gradient_bias**2)

            corrected_weights = first_moment_weights / (1 - beta1**step)
            corrected_weight_variance = second_moment_weights / (1 - beta2**step)
            corrected_bias = first_moment_bias / (1 - beta1**step)
            corrected_bias_variance = second_moment_bias / (1 - beta2**step)

            weights -= learning_rate * corrected_weights / (
                np.sqrt(corrected_weight_variance) + epsilon
            )
            bias -= learning_rate * corrected_bias / (np.sqrt(corrected_bias_variance) + epsilon)

    return SoftmaxModel(weights=weights, bias=bias, spec=spec)


def predict_probabilities(model: SoftmaxModel, features: np.ndarray) -> np.ndarray:
    logits = features @ model.weights + model.bias
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def metrics_from_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    accuracy = float((predictions == labels).mean())
    log_loss = float(
        -np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)))
    )
    return {"accuracy": accuracy, "log_loss": log_loss}


def run_model_experiment(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    numeric_columns: list[str],
    config: dict[str, float | int],
) -> tuple[SoftmaxModel, np.ndarray, dict[str, float]]:
    spec = fit_matrix_spec(train_frame, numeric_columns)
    train_features = build_design_matrix(train_frame, spec)
    test_features = build_design_matrix(test_frame, spec)
    train_labels = train_frame["y"].astype(int).to_numpy()
    test_labels = test_frame["y"].astype(int).to_numpy()

    model = fit_softmax_classifier(train_features, train_labels, spec, config)
    probabilities = predict_probabilities(model, test_features)
    return model, probabilities, metrics_from_probabilities(test_labels, probabilities)


def save_prediction_report(
    report_path: Path,
    test_frame: pd.DataFrame,
    model_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
) -> None:
    prediction_frame = test_frame[
        ["Season", "MatchDate", "HomeTeam", "AwayTeam", "FTR", "B365H", "B365D", "B365A"]
    ].copy()
    prediction_frame["MatchDate"] = prediction_frame["MatchDate"].dt.strftime("%Y-%m-%d")
    prediction_frame["ActualResult"] = prediction_frame["FTR"].map(RESULT_NAMES)
    prediction_frame["ModelPrediction"] = CLASS_LABELS[model_probabilities.argmax(axis=1)]
    prediction_frame["ModelPrediction"] = prediction_frame["ModelPrediction"].map(RESULT_NAMES)
    prediction_frame["MarketPrediction"] = CLASS_LABELS[market_probabilities.argmax(axis=1)]
    prediction_frame["MarketPrediction"] = prediction_frame["MarketPrediction"].map(RESULT_NAMES)
    prediction_frame["ModelHomeProb"] = model_probabilities[:, 0]
    prediction_frame["ModelDrawProb"] = model_probabilities[:, 1]
    prediction_frame["ModelAwayProb"] = model_probabilities[:, 2]
    prediction_frame["MarketHomeProb"] = market_probabilities[:, 0]
    prediction_frame["MarketDrawProb"] = market_probabilities[:, 1]
    prediction_frame["MarketAwayProb"] = market_probabilities[:, 2]
    prediction_frame.to_csv(report_path, index=False)


def save_model_artifact(path: Path, model: SoftmaxModel, trained_through: pd.Timestamp) -> None:
    np.savez(
        path,
        weights=model.weights,
        bias=model.bias,
        home_levels=np.array(model.spec.home_levels, dtype=str),
        away_levels=np.array(model.spec.away_levels, dtype=str),
        numeric_columns=np.array(model.spec.numeric_columns, dtype=str),
        numeric_means=model.spec.numeric_means,
        numeric_stds=model.spec.numeric_stds,
        class_labels=CLASS_LABELS.astype(str),
        trained_through=np.array([trained_through.strftime("%Y-%m-%d")], dtype=str),
    )


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_match_data()

    train_frame = matches[matches["SeasonStart"] < TEST_SEASON_START].copy()
    test_frame = matches[matches["Season"] == TEST_SEASON].copy()
    production_frame = matches.copy()

    if train_frame.empty or test_frame.empty:
        raise ValueError("Training or test split is empty. Check the available season files.")

    market_probabilities = test_frame[["prob_home", "prob_draw", "prob_away"]].to_numpy(
        dtype=np.float64
    )
    market_metrics = metrics_from_probabilities(test_frame["y"].astype(int).to_numpy(), market_probabilities)

    _, _, baseline_metrics = run_model_experiment(
        train_frame,
        test_frame,
        BASE_NUMERIC_COLUMNS,
        BASELINE_CONFIG,
    )
    _, odds_probabilities, odds_metrics = run_model_experiment(
        train_frame,
        test_frame,
        ODDS_NUMERIC_COLUMNS,
        ODDS_MODEL_CONFIG,
    )

    production_spec = fit_matrix_spec(production_frame, ODDS_NUMERIC_COLUMNS)
    production_features = build_design_matrix(production_frame, production_spec)
    production_labels = production_frame["y"].astype(int).to_numpy()
    production_model = fit_softmax_classifier(
        production_features,
        production_labels,
        production_spec,
        ODDS_MODEL_CONFIG,
    )

    predictions_path = OUTPUT_DIR / "prem_odds_predictions_2024_25.csv"
    metrics_path = OUTPUT_DIR / "prem_odds_metrics.json"
    model_path = OUTPUT_DIR / "prem_odds_softmax_model.npz"

    save_prediction_report(predictions_path, test_frame, odds_probabilities, market_probabilities)
    save_model_artifact(model_path, production_model, production_frame["MatchDate"].max())

    metrics_payload = {
        "data_directory": str(DATA_DIR),
        "matches_used": int(len(matches)),
        "seasons_used": int(matches["Season"].nunique()),
        "data_start": matches["MatchDate"].min().strftime("%Y-%m-%d"),
        "data_end": matches["MatchDate"].max().strftime("%Y-%m-%d"),
        "train_matches": int(len(train_frame)),
        "test_matches": int(len(test_frame)),
        "test_season": TEST_SEASON,
        "models": {
            "teams_only_softmax": baseline_metrics,
            "teams_plus_odds_softmax": odds_metrics,
            "market_implied_probabilities": market_metrics,
        },
        "notes": [
            "The training features are pre-match only: team identity, match date parts, and Bet365 1X2 odds.",
            "Post-match fields such as goals, shots, and cards are intentionally excluded to avoid target leakage.",
            "The saved production model is fit on every played match in the folder through the latest available date.",
        ],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(
        f"Loaded {len(matches)} matches from {metrics_payload['data_start']} to {metrics_payload['data_end']}."
    )
    print("Using pre-match features only. Post-match stats were excluded to avoid leaking the result.")
    print()
    print(f"Backtest on {TEST_SEASON}:")
    print(
        "  teams_only_softmax       "
        f"accuracy={baseline_metrics['accuracy']:.4f}  log_loss={baseline_metrics['log_loss']:.4f}"
    )
    print(
        "  teams_plus_odds_softmax  "
        f"accuracy={odds_metrics['accuracy']:.4f}  log_loss={odds_metrics['log_loss']:.4f}"
    )
    print(
        "  market_implied_probs     "
        f"accuracy={market_metrics['accuracy']:.4f}  log_loss={market_metrics['log_loss']:.4f}"
    )
    print()
    print(f"Saved prediction report: {predictions_path}")
    print(f"Saved metrics summary:  {metrics_path}")
    print(f"Saved model artifact:   {model_path}")


if __name__ == "__main__":
    main()
