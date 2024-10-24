#!/bin/bash

# Replace <p> with double-newline (note, HN doesn't add the closing </p> tag)
# Replace <i> or </i> with * for italics
# Repace multiple spaces with ' ' to remove repeated spaces
sed -e 's/  \+/ /g' -e 's/<p>/\n\n/g' -e 's/<\/\?i>/*/g'

