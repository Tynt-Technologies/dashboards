#!/bin/bash
echo "Activating venv"
source dashboard_env/bin/activate
echo "Running software"
python main_trd_dashboard.py
