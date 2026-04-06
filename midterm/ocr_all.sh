#!/usr/bin/env zsh

# Source folder (original PDFs)
src="Labs"
# Destination folder (OCR output)
dest="Labs_ocr"

# Create destination if it doesn't exist
mkdir -p "$dest"

# Find all PDF files under src and process them
find "$src" -type f -name "*.pdf" | while IFS= read -r file; do
    # Compute relative path from src
    rel="${file#$src/}"
    # Build output path
    out="$dest/$rel"
    # Create subdirectory in dest if needed
    out_dir=$(dirname "$out")
    mkdir -p "$out_dir"

    echo "Processing: $file -> $out"
    ocrmypdf --force-ocr -l eng "$file" "$out"
done

echo "All PDFs processed. Results are in $dest"
