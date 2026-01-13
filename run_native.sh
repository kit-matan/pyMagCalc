#!/bin/bash

# Simple launcher for pyMagCalc Native App

# Ensure we are running from the directory where the script is located
# This allows running the script from anywhere (e.g. via double click or from another dir)
cd "$(dirname "$0")"

echo "Starting pyMagCalc Native App..."
python gui/native_app.py
