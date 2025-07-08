#!/bin/bash
# Deployment script for risk-tool Docker setup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVER_USER="risk-tool"
SERVER_HOST="195.158.30.167"
SERVER_PATH="~/risk-tool"
LOCAL_DB_PATH="data/risk_tool.db"
LOCAL_ENV_PATH=".env"

echo -e "${GREEN}Risk Tool Docker Deployment Script${NC}"
echo "=================================="

# Step 1: Copy database and .env file to server
echo -e "\n${YELLOW}Step 1: Copying database and configuration files to server...${NC}"
echo "Copying database file..."
scp "$LOCAL_DB_PATH" "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/data/"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Database copied successfully${NC}"
else
    echo -e "${RED}✗ Failed to copy database${NC}"
    exit 1
fi

echo "Copying .env file..."
scp "$LOCAL_ENV_PATH" "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ .env file copied successfully${NC}"
else
    echo -e "${RED}✗ Failed to copy .env file${NC}"
    exit 1
fi

# Step 2: Execute deployment commands on server
echo -e "\n${YELLOW}Step 2: Deploying on server...${NC}"

ssh "${SERVER_USER}@${SERVER_HOST}" << 'ENDSSH'
set -e
cd ~/risk-tool

echo "Checking current Docker status..."
docker compose ps

echo "Stopping existing containers..."
docker compose down

echo "Pulling latest changes from git..."
git pull origin main

echo "Building Docker images..."
docker compose build --no-cache

echo "Starting services..."
docker compose up -d

echo "Waiting for services to start..."
sleep 10

echo "Checking container status..."
docker compose ps

echo "Verifying database connectivity..."
docker compose exec risk-tool python -c "import sqlite3; conn = sqlite3.connect('/app/data/risk_tool.db'); print('✓ Database connection successful'); conn.close()"

echo "Checking cron service..."
docker compose exec risk-tool ps aux | grep cron | grep -v grep && echo "✓ Cron service is running" || echo "✗ Cron service not found"

echo "Checking recent logs..."
docker compose logs --tail=20

echo "Dashboard URL (if accessible): http://195.158.30.167:8501"
ENDSSH

# Step 3: Test signal generation
echo -e "\n${YELLOW}Step 3: Testing signal generation...${NC}"
read -p "Do you want to test signal generation now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running signal generation test..."
    ssh "${SERVER_USER}@${SERVER_HOST}" << 'ENDSSH'
    cd ~/risk-tool
    docker compose exec risk-tool python send_daily_signals.py --save-only

    # Check if output file was created
    latest_file=$(docker compose exec risk-tool ls -t inference/outputs/ | head -1)
    if [ ! -z "$latest_file" ]; then
        echo "✓ Signal file generated: $latest_file"
        echo "Preview of generated file:"
        docker compose exec risk-tool head -50 "inference/outputs/$latest_file" | grep -E "(HIGH|MEDIUM|LOW|CRITICAL)" || echo "No risk levels found in preview"
    else
        echo "✗ No signal file generated"
    fi
ENDSSH
fi

echo -e "\n${GREEN}Deployment complete!${NC}"
echo "=================================="
echo "Next steps:"
echo "1. Monitor logs: ssh ${SERVER_USER}@${SERVER_HOST} 'cd ~/risk-tool && docker compose logs -f'"
echo "2. Check daily signals will be sent at 9:00 AM Tashkent time (4:00 AM UTC)"
echo "3. Database updates run at 8:30 AM Tashkent time (3:30 AM UTC)"
echo "4. Dashboard available at: http://195.158.30.167:8501 (if port is open)"
