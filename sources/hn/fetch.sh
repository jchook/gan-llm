#!/bin/bash

START_ID=4380000 # August 2012, when I signed-up for HN
FINAL_ID=30000000 # January 2022, the end of reliable human-generated text

set -e

# Resume from the last ID fetched
if [ -f data.json ]; then
  START_ID="$(tail -n 1 data.json | jq '.id')"
  let START_ID=START_ID+1
fi

# See the API docs: https://github.com/HackerNews/API
# IDs seem to be sequential, so we will just fetch them all
BASE_URL="https://hacker-news.firebaseio.com/v0/item"

# Fetch the data
while [ $START_ID -le $FINAL_ID ]; do
  curl --no-progress-meter "$BASE_URL/$START_ID.json" | tee -a data.json
  echo "" >> data.json
  let START_ID=START_ID+1
done

