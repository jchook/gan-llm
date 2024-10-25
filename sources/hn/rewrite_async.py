import os
import pandas as pd
import asyncio
import aiohttp
import json
from tqdm.asyncio import tqdm

# URL for your Ollama API
api_url = "http://localhost:11434/api/generate"

# Data files
input_csv_file = 'sources/hn/data/01-human.csv'
output_csv_file = 'sources/hn/data/01-machine.csv'

# Batch size for processing
batch_size = 3

# Define the prompt prefix for rewriting text
prompt_prefix = (
    "Rewrite this Hacker News comment in your own words. "
    "Do not plagiarize it. "
    "Only output your rewritten version of the supplied text. "
    "Do not add any extraneous preamble, footnotes, etc about it. "
    "Keep in mind that folks on Hacker News use the '>' prefix to quote other comments. "
    "Rewrite the quoted comment portions in the same manner as the comment itself. "
    "Okay, here is the original comment to rewrite: \n\n"
)

# Custom exception for rewriting errors
class RewritingError(Exception):
    pass

# Async function to rewrite the comments using the Ollama API
async def rewrite_essay(session, text, model_name="llama3.2"):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({
        "model": model_name,
        "prompt": f"{prompt_prefix}{text}",
        "stream": False
    })
    async with session.post(api_url, data=payload, headers=headers) as response:
        if response.status == 200:
            result = await response.json()
            return result.get('response', None)
        else:
            raise RewritingError(f"Error: {response.status}, {await response.text()}")

# Async function to rewrite essays in batches
async def rewrite_in_batches(data, start_index, batch_size=batch_size):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index in range(start_index, min(start_index + batch_size, len(data))):
            row = data.iloc[index]
            tasks.append(rewrite_essay(session, row['text']))
        return await asyncio.gather(*tasks)

# Main function to handle rewriting with batch processing
async def main():
    # Load the cleaned data
    data = pd.read_csv(input_csv_file)

    # Open the file in append mode
    with open(output_csv_file, 'a', newline='') as f:
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

        # Calculate the total number of rows to be written
        total_rows = len(data) - num_rows_written

        # Iterate over the rows in batches
        for start_index in tqdm(range(num_rows_written, len(data), batch_size), total=total_rows // batch_size, desc="Rewriting data"):
            batch_data = data.iloc[start_index:start_index + batch_size]
            try:
                rewritten_texts = await rewrite_in_batches(data, start_index, batch_size=batch_size)
                # Write the rewritten data to CSV
                for idx, rewritten_text in enumerate(rewritten_texts):
                    row = batch_data.iloc[idx]
                    df = pd.DataFrame({
                        'id': [row['id']],
                        'text': [rewritten_text],
                        'label': [1],
                    })
                    df.to_csv(f, mode='a', header=False, index=False)
            except RewritingError as e:
                print(f"Skipping batch due to an error: {e}")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

