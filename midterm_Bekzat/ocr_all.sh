#!/usr/bin/env zsh

set -euo pipefail

source_dir="${1:-Labs}"
output_dir="${2:-Labs_ocr}"

if [[ ! -d "$source_dir" ]]; then
    print -u2 "Source directory not found: $source_dir"
    exit 1
fi

mkdir -p "$output_dir"

find "$source_dir" -type f -name "*.pdf" -print0 | while IFS= read -r -d '' pdf_file; do
    relative_path="${pdf_file#$source_dir/}"
    target_file="$output_dir/$relative_path"
    mkdir -p "${target_file:h}"
    print "OCR: $pdf_file -> $target_file"
    ocrmypdf --force-ocr -l eng "$pdf_file" "$target_file"
done

print "OCR output written to $output_dir"
