import pandas as pd
from openai import OpenAI

client = OpenAI()

_SYSTEM_PROMPT = (
    "Assume the role of Hacker News comment rewriter. Rewrite this Hacker News "
    "comment in your own words. Try to preserve the author's writing style and "
    "essence without plagiarizing it or keeping the exact same word choice. Try "
    "to capture the author's writing quirks such as innocuous misspellings, "
    "spacing, punctuation preferences, sentence structure, mood, "
    "formality/informality, attitude, etc. Only output the rewritten comment. "
    "Here is the comment to rewrite: \n\n"
)

def rephrase_text_with_gpt4(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that rephrases text."},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0]['message']['content']

def main():
    # Read the human.csv file
    df = pd.read_csv('human.csv')

    # Initialize a list to store the new rows
    new_data = []

    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        original_text = row['text']
        rephrased_text = rephrase_text_with_gpt4(original_text)
        new_label = 1  # Change label from 0 to 1

        # Append the new row to the list
        new_data.append({'id': row['id'], 'text': rephrased_text, 'label': new_label})

    # Create a new dataframe from the new data
    new_df = pd.DataFrame(new_data)

    # Save the new dataframe to machine.csv
    new_df.to_csv('machine.csv', index=False)

if __name__ == "__main__":
    main()

