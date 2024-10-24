#!/bin/bash
# Filter json data to only include entries where:
# - the "type" field is "comment"
# - the "text" field is at least 500 characters long
# - the "text" field does not contain the string "href"
jq -c 'select(.type == "comment" and (.text | length) >= 500 and (.text | test("href") | not))'

