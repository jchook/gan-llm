"""
This script processes the Hacker News dataset to generate a training dataset
for the conversation model. It pairs comments with their parent comments, and
is intended to train chat-like models to generate responses to comments similar
to those found on Hacker News.

The data is output in ShareGPT format, which can be loaded into Unsloth for
masked language model fine-tuning.
"""
import json
import typer
from tqdm import tqdm

def main(
    input_file="sources/hn/data/gen/filtered.jsonl",
    output_file="sources/hn/data/gen/train.json"
):
    parent_comments = {}
    output_data = []

    # First pass to identify all comments with kids (parents)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading parents"):
            line = line.strip()
            if not line:
                continue
            comment = json.loads(line)
            if "kids" in comment:
                parent_comments[comment["id"]] = comment

    # Second pass to match kids with their parents and structure data
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing comments"):
            line = line.strip()
            if not line:
                continue
            comment = json.loads(line)
            parent_id = comment.get("parent")

            # If this comment is a "kid" of a parent comment
            if parent_id in parent_comments:
                parent_text = parent_comments[parent_id]["text"]
                kid_text = comment["text"]

                # Append to output format
                output_data.append({"conversations":[
                    {"from": "human", "value": parent_text},
                    {"from": "gpt", "value": kid_text}
                ]})

    # Write output data to file in JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(output_data) + '\n')

if __name__ == "__main__":
    typer.run(main)

