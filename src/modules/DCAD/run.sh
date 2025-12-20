#!/bin/bash
# Helper script to run app.py with the virtual environment

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the app
python3 app.py "$@"
