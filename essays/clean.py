import re
import pandas as pd
from faker import Faker
import random

fake = Faker()

def fake_date():
    random_date = fake.date_between(start_date="-10y", end_date="today")  # Generates a random date within the past 10 years
    return random_date.strftime("%B %d, %Y")

def fake_phone_number():
    phone_number = fake.phone_number()
    clean_number = re.sub(r'x\d+', '', phone_number).strip()
    return clean_number

replacements = {
    r"@PERSON\d+": fake.name,
    r"@ORGANIZATION\d+": fake.company,
    r"@LOCATION\d+": fake.city,
    r"@DATE\d+": fake_date,
    r"@TIME\d+": fake.time,
    r"@MONEY\d+": fake.pricetag,  # Or fake.currency + fake.random_number for formatting it yourself
    r"@PERCENT\d+": lambda: f"{fake.random_number(digits=2)}%",
    r"@NUM\d+": lambda: f"{fake.random_number(digits=1)}",
    #r"@NUM\d+": fake_phone_number,
    r"@EMAIL\d+": fake.email,
    r"@MONTH\d+": fake.month_name,
    r"@CAPS\d+": fake.word,  # For now, could be enhanced depending on the need
    r"@DR\d+": lambda: f"Dr. {fake.last_name()}",
    r"@CITY\d+": fake.city,
    r"@STATE\d+": fake.state
}

def clean(text, replacements):
    replacement_memory = {}
    for placeholder_pattern, generator in replacements.items():
        matches = re.findall(placeholder_pattern, text)
        for match in matches:
            if match not in replacement_memory:
                replacement = generator()
                if placeholder_pattern == r"@PERSON\d+":
                    name_parts = replacement.split(" ")
                    if len(name_parts) > 1:
                        replacement_memory[match] = random.choice([name_parts[0], name_parts[1]])
                    else:
                        replacement_memory[match] = replacement
                else:
                    replacement_memory[match] = replacement
            else:
                replacement = replacement_memory[match]
            text = text.replace(match, replacement, 1)
    return text

def coerce_to_utf8(text):
    return str(text).encode('utf-8', errors='ignore').decode('utf-8')

documents = pd.read_csv(
    "./datasets/asap-aes/training_set_rel3.tsv",
    sep="\t",
    encoding='ISO-8859-1'
)

reformatted_documents = documents[['essay_id', 'essay']].copy()
reformatted_documents['origin'] = 0
reformatted_documents['essay'] = reformatted_documents['essay'].apply(coerce_to_utf8)

for index, row in documents.iterrows():
    reformatted_documents.at[index, 'essay'] = clean(row['essay'], replacements)

reformatted_documents.to_csv('essays.csv', index=False)
