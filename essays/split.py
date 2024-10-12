import pandas as pd
from sklearn.model_selection import train_test_split

# File paths for the CSVs you want to concatenate
csv_files = ["essays/essays.csv", "essays/rewritten_essays.csv"]

# Concatenate the CSV files into a single DataFrame
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Split the data into train (80%), test (10%), and validation (10%) sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# Output the splits into separate CSV files
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
val_df.to_csv('data/validation.csv', index=False)

print(f"Data successfully split: {len(train_df)} training rows, {len(test_df)} testing rows, {len(val_df)} validation rows.")

