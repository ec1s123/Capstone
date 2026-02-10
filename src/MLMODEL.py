import pandas as pd

df = pd.read_csv(r'C:\Users\Player 1\Documents\Capstone\src\data\epl_final.csv')
print(df.head(5))

missing  = df.isnull().sum()
print(" Missing values in each column:\n", missing)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print("Numeric columns in the dataset:\n", numeric_cols)

catergory_cols = df.select_dtypes(include=['object']).columns
print("Catergory columns in the dataset:\n", catergory_cols)

# Encoding
# Parse MatchDate into numeric features 
df['MatchDate'] = pd.to_datetime(df['MatchDate'])
df['MatchYear'] = df['MatchDate'].dt.year
df['MatchMonth'] = df['MatchDate'].dt.month
df['MatchDayOfWeek'] = df['MatchDate'].dt.dayofweek

# Label-encode target column 
label_map = {'H': 0, 'D': 1, 'A': 2}
df['y'] = df['FullTimeResult'].map(label_map)
df = df.drop(columns=['FullTimeResult'])

print("Encoded dataset preview:\n", df.head(-5))

# All the years before the test season
train = df[df["Season"] < "2023/24"]
# Most recent complete season for testing
test  = df[df["Season"] == "2024/25"]

print("Training set shape:", train.shape) 
print("Testing set shape:", test.shape)

print(df.columns)