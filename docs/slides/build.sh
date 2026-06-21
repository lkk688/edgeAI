#!/usr/bin/env bash
# Build every Marp deck in this folder:  docs/slides/*.md  ->  docs/slides/*.html
# The .html is published by `mkdocs gh-deploy`; the .md sources are excluded from
# the MkDocs build (see `exclude_docs` in mkdocs.yml).
#   ./docs/slides/build.sh           # build HTML
#   ./docs/slides/build.sh --pdf     # also export PDF handouts
set -e
cd "$(dirname "$0")"                  # docs/slides
MARP="npx --yes @marp-team/marp-cli@latest"
for f in *.md; do
  [ "$f" = "README.md" ] && continue
  name="${f%.md}"
  echo "• $f -> $name.html"
  $MARP "$f" --html -o "$name.html"
  if [ "${1:-}" = "--pdf" ]; then
    $MARP "$f" --html --allow-local-files --pdf -o "$name.pdf"
  fi
done
echo "✅ Built decks. Preview the .html locally or publish with: mkdocs gh-deploy"
