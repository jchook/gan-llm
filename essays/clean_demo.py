# Function to process multiple documents
def process_documents(documents, replacements):
    processed_documents = []

    for doc in documents:
        processed_doc = replace_placeholders_in_document(doc, replacements)
        processed_documents.append(processed_doc)

    return processed_documents

# Example usage
documents = [
    """
    @PERSON1 loves @PERSON2. That's because @PERSON2 is a sweetie. They visited @LOCATION1.
    """,
    """
    @PERSON1 and @PERSON2 went to @LOCATION1 on @DATE1. @PERSON1 was very happy.
    """,
    """
    Yesterday, @PERSON1 met with @PERSON2 at @ORGANIZATION1 to discuss their new project. They decided to launch it on @DATE1 at @TIME1. The project budget is estimated to be around @MONEY1, and they expect a growth rate of @PERCENT1 in the first year.

    After the meeting, @PERSON1 and @PERSON2 went to @LOCATION1 for lunch. They ordered a meal for @MONEY2 and exchanged emails: @PERSON1 can be reached at @EMAIL1 and @PERSON2 at @EMAIL2.

    Later that day, they drove through @CITY1, passing by @STATE1, and ended up at @PERSON3's house. @PERSON3, also known as @DR1, had some great stories to tell. The trio reminisced about their trip last @MONTH1, where they explored various parts of @CITY2 and enjoyed the local culture.

    As @PERSON1 left @PERSON3's house, they realized they had forgotten their phone number, which is @NUM1. Hopefully, they'll retrieve it when they meet again on @DATE2.
    """
]

# Process all documents
processed_documents = process_documents(documents, replacements)

# Print results
for idx, doc in enumerate(processed_documents, 1):
    print(f"Document {idx}:\n{doc}\n")

