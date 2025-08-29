#!/bin/bash
# Setup script for daily risk pipeline cron job

echo "Setting up daily risk pipeline automation..."

# Get the current directory
PROJECT_DIR=$(pwd)
PYTHON_PATH=$(which python)

# Create the cron job command
CRON_COMMAND="0 7 * * 1-5 cd $PROJECT_DIR && $PYTHON_PATH morning_pipeline.py >> logs/morning_pipeline.log 2>&1"

# Create logs directory if it doesn't exist
mkdir -p logs

# Add to crontab (runs at 7 AM Monday-Friday)
echo "Adding cron job: $CRON_COMMAND"
(crontab -l 2>/dev/null; echo "$CRON_COMMAND") | crontab -

echo "✅ Cron job set up successfully!"
echo "The risk pipeline will run at 7 AM every weekday"
echo "Logs will be saved to: $PROJECT_DIR/logs/morning_pipeline.log"

# Show current crontab
echo -e "\nCurrent cron jobs:"
crontab -l
