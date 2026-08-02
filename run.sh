#!/bin/bash
# macOS/Linux one-command startup
set -e

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt --quiet
uvicorn main:app --reload --host 0.0.0.0 --port 8000
