import csv
import html
import json
import re
import typer

def check_text(text):
    return (
        text is not None
        and re.search(r'<a href=|<code>', text) is None
        and len(text) >= 500
    )


def process_text(text):
    # Convert <p> to line breaks (note: HN doesn't close <p> tags)
    text = re.sub(r'<p>', '\n', text)

    # Remove HTML tags
    text = re.sub(r'<.+?>', '', text)

    # Convert HTML entities
    text = html.unescape(text)

    # Remove extra whitespaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Consolidate line breaks
    text = re.sub(r'\n+', '\n\n', text)

    # Ok
    return text


def main(input_json_file='sources/hn/data/02/human.json', output_csv_file='sources/hn/data/02/human.csv'):
    with (
        open(input_json_file, 'r') as json_file,
        open(output_csv_file, 'w', newline='') as csv_file
    ):
        # Set up CSV writer
        csv_writer = csv.writer(csv_file)

        # Write the header row for CSV
        csv_writer.writerow(['id', 'text', 'label'])

        # Process each line in the JSON file
        for line in json_file:
            try:
                # Parse the JSON line
                data = json.loads(line)

                # Extract the required fields
                id_value = data.get('id')
                text_value = data.get('text')

                # Only look at comments
                if data.get('type') != 'comment':
                    continue

                # ID is required
                if id_value is None:
                    continue

                # Skip lines that do not meet criteria
                if not check_text(text_value):
                    continue

                # Process the text
                text_value = process_text(text_value)

                # Write the row to the CSV file
                csv_writer.writerow([id_value, text_value, 0])

            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")

    print(f"Conversion complete! CSV saved to: {output_csv_file}")


if __name__ == "__main__":
    typer.run(main)
