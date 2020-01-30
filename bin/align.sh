#!/usr/bin/env bash
approot=$(cd "$(dirname "$0")"/.. && pwd)
source "$approot/venv/bin/activate"
python "$approot/align/align.py" "$@"
stty sane
