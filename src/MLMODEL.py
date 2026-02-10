import pandas as pd
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score


df = pd.read_csv(r'/Users/adam/Documents/Capstone/Capstone/src/data/epl_final.csv')
print(df.head(5))

missing  = df.isnull().sum()
print(" Missing values in each column:\n", missing)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print("Numeric columns in the dataset:\n", numeric_cols)

catergory_cols = df.select_dtypes(include=['object']).columns
print("Catergory columns in the dataset:\n", catergory_cols)

# Encoding

# Label-encode target column
label_map = {'H': 0, 'D': 1, 'A': 2}
df['y'] = df['FullTimeResult'].map(label_map)
df = df.drop(columns=['FullTimeResult'])

print("Encoded dataset preview:\n", df.head(-5))

# One-hot encode categorical feature columns (excluding split-only columns)
feature_cat_cols = [c for c in catergory_cols if c not in ['FullTimeResult', 'Season', 'MatchDate']]
if feature_cat_cols:
    df = pd.get_dummies(df, columns=feature_cat_cols)

# All the years before the test season
train = df[df["Season"] < "2023/24"]
# Most recent complete season for testing
test  = df[df["Season"] == "2024/25"]

print("Training set shape:", train.shape) 
print("Testing set shape:", test.shape)

print(df.columns)

# Modeling
X_train = train.drop(columns=['y', 'Season', 'MatchDate'])
y_train = train['y']
X_val = test.drop(columns=['y', 'Season', 'MatchDate'])
y_val = test['y']


model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

probs = model.predict_proba(X_val)
preds = probs.argmax(axis=1)

print("Accuracy:", accuracy_score(y_val, preds))
print("Log Loss:", log_loss(y_val, probs))
