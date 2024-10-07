import pandas as pd
import requests
import json
from tqdm import tqdm

# URL for your Ollama API
api_url = "http://localhost:11434/api/generate"

# Define the prompt prefix for rewriting essays
prompt_prefix = (
    "Completely rewrite this essay in your own words. Do not plaigairize it. "
    "Only output your version of the essay. "
    "Do not make any footnotes or comments about it."
)

# Custom exception for rewriting errors
class RewritingError(Exception):
    pass

# Rewrite the essays using the API
def rewrite_essay(essay_text, model_name="llama3.2"):

    # Define the API request payload
    payload = {
        "model": model_name,
        "prompt": f"{prompt_prefix} {essay_text}",
        "stream": False
    }

    # Send the POST request to the API
    response = requests.post(api_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

    # Check for a successful response
    if response.status_code == 200:
        return response.json().get('response', None)
    else:
        raise RewritingError(f"Error: {response.status_code}, {response.text}")

# Load your cleaned essays
essays = pd.read_csv('essays.csv')

# Define the CSV file path
csv_file = 'rewritten_essays.csv'

# Open the file in append mode once
with open(csv_file, 'a', newline='') as f:
    # Write the header initially
    f.write("essay_id,essay\n")

    # Create a CSV writer object and progressively append rows
    for index, row in tqdm(essays.iterrows(), total=len(essays), desc="Rewriting Essays"):
        essay_id = row['essay_id']
        essay_text = row['essay']

        # Call your existing function to rewrite the essay
        rewritten_text = rewrite_essay(essay_text)

        # Create a DataFrame for the current row
        df = pd.DataFrame({
            'essay_id': [essay_id],
            'essay': [rewritten_text]
        })

        # Append the DataFrame to the already-opened file
        df.to_csv(f, mode='a', header=False, index=False)
