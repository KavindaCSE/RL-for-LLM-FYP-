#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the main script to start the training and evaluation process
python src/main.py --config experiments/first_approach/config.yaml

# Deactivate the virtual environment
deactivate