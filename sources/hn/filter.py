"""
This script processes Hacker News JSONL data files, filters out unwanted data,
converts the text to a suitable format, and outputs the data in the same JSONL
format, ready for further processing.

Unlike filter.sh, this script will also reformat the text data to remove HTML
tags, convert HTML entities, and consolidate line breaks.
"""
import json
import html
import re
import typer
from tqdm import tqdm

def check_text(text, min_length=500):
    """
    Check if the text is valid for training data.

    In this case, we will lower the threshold to 200 characters. We want
    sufficiently long responses to ensure they are distinguishable from human
    written output, but not as long as the discriminator fine-tuning data.
    """
    return (
        text is not None
        and re.search(r'<a href=|<code>', text) is None
        and len(text) >= min_length
    )

def process_text(text):
    # Convert <p> to line breaks (note: HN doesn't close <p> tags)
    text = re.sub(r'<p>', '\n', text)

    # Remove HTML tags
    # Note, this method is not "safe" for XSS, etc, but good enough for HN data
    text = re.sub(r'<.+?>', '', text)

    # Convert HTML entities
    text = html.unescape(text)

    # Remove extra whitespaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Consolidate line breaks
    text = re.sub(r'\n+', '\n\n', text)

    # Ok
    return text


def main(
    input_file="sources/hn/data/gen/1m.jsonl",
    output_file="sources/hn/data/gen/filtered.jsonl",
    min_length:int=200
):
    with (
        open(input_file, 'r') as input_fp,
        open(output_file, 'w', newline='') as output_fp
    ):
        # Process each line in the JSON file
        for line in tqdm(input_fp):
            try:
                if not line.strip(): continue

                # Parse the JSON line
                data = json.loads(line)

                # Only look at comments
                if (
                    data.get("type") != 'comment'
                    or data.get("id") is None
                    or data.get("text") is None
                    or not check_text(data["text"], min_length)
                ): continue

                # Process the text
                data["text"] = process_text(data["text"])

                # Write the processed data to the output file
                output_fp.write(json.dumps(data) + "\n")

            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")

    print(f"Conversion complete! Data saved to: {output_file}")


if __name__ == "__main__":
    typer.run(main)


