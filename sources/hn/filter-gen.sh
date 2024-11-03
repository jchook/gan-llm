#!/bin/bash
# Filter json data to only include entries where:
# - the document does not contain the string "<a href=" or "<code>"
# - the "type" field is "comment"
# - the "text" field is at least 200 characters long
grep -v -e '<a href=' -e '<code>' | \
  jq -c 'select(.type == "comment" and (.text | length) >= 200)'


