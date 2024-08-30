#!/bin/bash

# The directory containing subfolders with test.csv files.
# Replace "/path/to/your/folder" with the actual folder path you will provide.
PYTHON_SCRIPT_PATH="/Users/emmazhuang/Documents/Codes/Masakhane/openai/gpt_mgsm.py"

# Path to your Python script.
# Make sure this path points to where openai_model5.py is located.
FOLDER_PATH="/Users/emmazhuang/Documents/Codes/Masakhane/afrimgsm"

# Iterate over each subfolder in the provided directory.
find "${FOLDER_PATH}" -type d | while read -r SUBFOLDER; do
  # Define the path to the test.csv file within the current subfolder.
  INPUT_FILE="${SUBFOLDER}/test.tsv"
  language=$(echo "$SUBFOLDER" | cut -d'/' -f8)
  
  # Define the path for the output.csv file within the current subfolder.
  OUTPUT_FILE="/Users/emmazhuang/Documents/Codes/Masakhane/afrimgsm/gpt4/${language}.csv"
  
  # Check if the test.csv file exists in the subfolder.
  if [[ -f "${INPUT_FILE}" ]]; then
    # Run the Python script with the input and output file paths.
    echo "Processing ${INPUT_FILE}..."
    python3 "${PYTHON_SCRIPT_PATH}" "${INPUT_FILE}" "${OUTPUT_FILE}"
    echo "Output saved to ${OUTPUT_FILE}"
  else
    echo "test.csv not found in ${SUBFOLDER}"
  fi
done