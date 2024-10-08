import pandas as pd
import os

# Define the file path
csv_file_path = "essays/rewritten_essays.csv"

# Check if the file exists
if os.path.exists(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Append a new column called 'origin' with the value 1 for every row
    df['origin'] = 1

    # Write the updated dataframe back to the CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"Successfully added 'origin' column to {csv_file_path}")
else:
    print(f"File not found: {csv_file_path}. Please check the file path.")

