#!/bin/bash

# Check if the version number argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <version_folder_number>"
  exit 1
fi

VERSION_NUMBER=$1
SOURCE_FOLDER="logs/OnlineSpatialNet/version_${VERSION_NUMBER}"
DESTINATION_FOLDER="myExpLogs/logs/OnlineSpatialNet/version_${VERSION_NUMBER}"

# Check if the source folder exists
if [ ! -d "$SOURCE_FOLDER" ]; then
  echo "Source folder $SOURCE_FOLDER does not exist."
  exit 1
fi

# Create the destination folder with the same structure as the source
mkdir -p "$DESTINATION_FOLDER"

# Copy all files except the checkpoints folder while preserving the structure
rsync -av --exclude='checkpoints' "$SOURCE_FOLDER/" "$DESTINATION_FOLDER/"

echo "Files from $SOURCE_FOLDER copied to $DESTINATION_FOLDER successfully."
