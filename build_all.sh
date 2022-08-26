#!/bin/bash
set -e
find . -name '*.md' | xargs -I{} bash -c 'python build.py {} --pandoc; echo {}'
cd teach/csc411_19s
python build.py index.md --pandoc
