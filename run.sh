#!/usr/bin/env bash
# Usage: ./run.sh <audio_folder> [output_folder]
set -e
python run.py --input "${1:?audio folder required}" --output "${2:-outputs}"
