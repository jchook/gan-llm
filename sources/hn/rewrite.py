import os
import pandas as pd
import requests
import json
from tqdm import tqdm

# URL for your Ollama API
api_url = "http://localhost:11434/api/generate"

# Data files
input_csv_file = 'sources/hn/data/01-human.csv'
output_csv_file = 'sources/hn/data/01-machine.csv'


# Define the prompt prefix for rewriting text
prompt_prefix = (
    "Rewrite this Hacker News comment in your own words. "
    "Do not plaigairize it. "
    "Only output your rewritten version of the supplied text. "
    "Do not add any extraneous preamble, footnotes, etc about it. "
    "Keep in mind that folks on Hacker News use the '>' prefix to quote other comments. "
    "Rewrite the quoted comment portions in the same manner as the comment itself. "
    "Okay, here is the original comment to rewrite: \n\n"
)

# Custom exception for rewriting errors
class RewritingError(Exception):
    pass

# Rewrite the comments using the Ollama API
def rewrite_essay(text, model_name="llama3.2"):
    headers = { 'Content-Type': 'application/json' }
    payload = json.dumps({
        "model": model_name,
        "prompt": f"{prompt_prefix}{text}",
        "stream": False
    })
    response = requests.post(api_url, data=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get('response', None)
    else:
        raise RewritingError(f"Error: {response.status_code}, {response.text}")

# Load the cleaned data
data = pd.read_csv(input_csv_file)

# Open the file in append mode once
with open(output_csv_file, 'a', newline='') as f:
    rows = data.iterrows()
    if os.stat(output_csv_file).st_size == 0:
        # Write the header row only if the file is empty
        print("New CSV file. Writing the header row...")
        f.write("id,text,label\n")
        num_rows_written = 0
    else:
        # To resume rewriting (since this is a time-intensive task),
        # read the number of rows already written to the file
        df = pd.read_csv(output_csv_file)
        num_rows_written = len(df)
        del df
        print(f"Resuming rewriting from row {num_rows_written}...")

        # Skip rows that have already been written
        row_num = 0
        for _, _ in rows:
            row_num += 1
            if row_num >= num_rows_written:
                break


    # Calculate the total number of rows to be written
    total_rows = len(data) - num_rows_written

    # Create a CSV writer object and progressively append rows
    for index, row in tqdm(rows, total=total_rows, desc="Rewriting data"):

        # Extract the essay ID and text
        id = row['id']
        text = row['text']

        # Rewrite the essay
        rewritten_text = rewrite_essay(text)

        # Create a DataFrame for the current row
        df = pd.DataFrame({
            'id': [id],
            'text': [rewritten_text],
            'label': [1],
        })

        # Append the DataFrame to the already-opened file
        df.to_csv(f, mode='a', header=False, index=False)

