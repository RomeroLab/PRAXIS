#!/usr/bin/env python3
"""Strip outputs and volatile metadata from a Jupyter notebook.

Reads a notebook on stdin, writes the stripped notebook to stdout. Used as a git
clean filter, so the working copy keeps its rendered figures while the committed
version stores only source. That keeps notebook diffs readable and stops
rendered PNGs from dominating the repository.

Set up once per clone (paths are relative to the repository root, which is where
git runs filter commands from):

    git config filter.nbstrip.clean "python3 paper_analysis/tools/nbstrip.py"
    git config filter.nbstrip.smudge cat
    git config filter.nbstrip.required true

.gitattributes already routes *.ipynb through it.
"""
import json
import sys

# Per-cell metadata that changes on every run and carries no information.
VOLATILE_CELL_METADATA = ('execution', 'collapsed', 'scrolled', 'ExecuteTime')


def strip(nb):
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        meta = cell.get('metadata', {})
        for key in VOLATILE_CELL_METADATA:
            meta.pop(key, None)
    # A pinned interpreter version churns whenever the kernel is upgraded.
    nb.get('metadata', {}).get('language_info', {}).pop('version', None)
    return nb


def main():
    raw = sys.stdin.read()
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        # Not a notebook we can parse — pass it through untouched rather than
        # risk corrupting whatever this file actually is.
        sys.stdout.write(raw)
        return
    json.dump(strip(nb), sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
