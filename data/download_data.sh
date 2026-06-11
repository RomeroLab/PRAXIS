#!/bin/bash
#
# Restore the large artifacts that are not committed to git:
#   - benchmark DMS oracle datasets (*_SeqFxnDataset.pkl) into agent/insilico_analysis/
#   - (model weights ESM2-650M + Tranception are downloaded separately by agent/setup.sh)
#
# Usage:
#   bash data/download_data.sh
#
# Override the archive location if needed:
#   PRAXIS_DATA_URL=https://zenodo.org/records/<RECORD_ID>/files bash data/download_data.sh

set -euo pipefail

# TODO: replace with the published Zenodo record base URL once the dataset DOI is minted.
BASE_URL="${PRAXIS_DATA_URL:-https://zenodo.org/records/REPLACE_ME/files}"

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
MANIFEST="$REPO_ROOT/data/benchmark_data_manifest.txt"

echo "Restoring benchmark oracle datasets from: $BASE_URL"
echo ""

while read -r rel_path bytes _; do
  case "$rel_path" in
    ''|\#*) continue ;;   # skip blanks and comments
  esac
  dest="$REPO_ROOT/$rel_path"
  fname="$(basename "$rel_path")"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    echo "  ✓ already present: $rel_path"
    continue
  fi
  echo "  ↓ $fname  ($bytes bytes) -> $rel_path"
  curl -fSL "$BASE_URL/$fname" -o "$dest"
done < "$MANIFEST"

echo ""
echo "Done. For model weights (ESM2-650M + Tranception Large), run: bash agent/setup.sh"
