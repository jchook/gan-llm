#!/bin/bash

START_ID=20000000 # May 2019
FINAL_ID=22000000 # January 2020
OUT_FILE="data/data.json"

set -e

# Resume from the last ID fetched
if [ -f "$OUT_FILE" ]; then
  START_ID="$(tail -n 1 "$OUT_FILE" | jq '.id')"
  let START_ID=START_ID+1
fi

# See the API docs: https://github.com/HackerNews/API
# IDs seem to be sequential, so we will just fetch them all
BASE_URL="https://hacker-news.firebaseio.com/v0/item"

# Fetch the data
while [ $START_ID -le $FINAL_ID ]; do
  curl --no-progress-meter "$BASE_URL/$START_ID.json" | tee -a "$OUT_FILE"
  echo "" >> "$OUT_FILE"
  let START_ID=START_ID+1
done

