#!/bin/bash
set -e

# Ensure log directory exists with correct permissions
mkdir -p /app/logs /app/data /app/inference/outputs

# Initialize cron log
touch /app/logs/cron.log

# Test database connection
echo "Testing database connection..."
python -c "import sqlite3; conn = sqlite3.connect('/app/data/risk_tool.db'); print('Database connection successful'); conn.close()"

# If running cron (default), start cron daemon
if [ "$1" = "cron" ]; then
    echo "Starting cron daemon for daily automation..."
    # Start cron in foreground
    exec cron -f
else
    # Otherwise, execute the command
    exec "$@"
fi
