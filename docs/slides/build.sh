#!/usr/bin/env bash
# Build every Marp deck in the specified directory or current directory and its subdirectories.
# Usage:
#   ./build.sh [directories...] [--pdf]
# E.g.:
#   ./build.sh                      # build HTML recursively under the slides directory
#   ./build.sh --pdf                # build HTML & PDF recursively under the slides directory
#   ./build.sh autonomous --pdf     # build HTML & PDF under autonomous/

set -e

# Resolve the absolute path of the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MARP="npx --yes @marp-team/marp-cli@latest"

PDF_EXPORT=false
DIRS=()

for arg in "$@"; do
  if [ "$arg" = "--pdf" ]; then
    PDF_EXPORT=true
  else
    DIRS+=("$arg")
  fi
done

# If no directories specified, default to the script's directory
if [ ${#DIRS[@]} -eq 0 ]; then
  DIRS=("$SCRIPT_DIR")
fi

for dir in "${DIRS[@]}"; do
  echo "🔍 Searching for slides in: $dir"
  if [ ! -d "$dir" ]; then
    echo "⚠️ Warning: Directory '$dir' does not exist, skipping."
    continue
  fi
  
  # Find all .md files under $dir, excluding README.md and themes
  find "$dir" -name "*.md" ! -name "README.md" ! -path "*/themes/*" ! -path "*/node_modules/*" | while read -r f; do
    # Resolve absolute path of markdown file
    f_abs="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
    
    # Calculate output name
    name="${f_abs%.md}"
    
    echo "• Building: $f -> ${name}.html"
    $MARP "$f_abs" --html --no-stdin \
      --theme-set "$SCRIPT_DIR/themes/sjsu.css" \
      --theme-set "$SCRIPT_DIR/themes/autonomous.css" \
      -o "${name}.html"
      
    if [ "$PDF_EXPORT" = true ]; then
      echo "• Exporting PDF: ${name}.pdf"
      $MARP "$f_abs" --html --no-stdin --allow-local-files \
        --theme-set "$SCRIPT_DIR/themes/sjsu.css" \
        --theme-set "$SCRIPT_DIR/themes/autonomous.css" \
        --pdf -o "${name}.pdf"
    fi
  done
done

echo "✅ Built all requested decks."
