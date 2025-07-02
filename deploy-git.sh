#!/bin/bash
# Git-based deployment script for Risk Tool
# Usage: ./deploy-git.sh [--initial-setup]

set -e

# Configuration
REMOTE_USER="risk-tool"
REMOTE_HOST="195.158.30.167"
REMOTE_DIR="/home/risk-tool/risk-tool"
REPO_URL="git@github.com:xtemur/risk-tool.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
INITIAL_SETUP=false
if [[ "$1" == "--initial-setup" ]]; then
    INITIAL_SETUP=true
fi

echo -e "${GREEN}=== Risk Tool Git Deployment ===${NC}"

if [ "$INITIAL_SETUP" = true ]; then
    echo -e "${YELLOW}Running initial setup...${NC}"

    # Step 1: Commit and push local changes
    echo -e "${YELLOW}Step 1: Preparing Git repository...${NC}"
    git add -A
    git commit -m "Initial deployment setup $(date +%Y-%m-%d_%H:%M:%S)" || echo "No changes to commit"

    # Step 2: Initial setup on remote server
    echo -e "${YELLOW}Step 2: Setting up remote server...${NC}"
    ssh "$REMOTE_USER@$REMOTE_HOST" << EOF
set -e

# Remove existing directory if it exists
if [ -d "$REMOTE_DIR" ]; then
    echo "Backing up existing installation..."
    mv $REMOTE_DIR ${REMOTE_DIR}_backup_\$(date +%Y%m%d_%H%M%S)
fi

# Clone repository
echo "Cloning repository..."
git clone $REPO_URL $REMOTE_DIR
cd $REMOTE_DIR

# Copy environment file from backup if it exists
if [ -f "${REMOTE_DIR}_backup_*/".env.docker ]; then
    echo "Restoring previous environment configuration..."
    cp ${REMOTE_DIR}_backup_*/.env.docker .env.docker
else
    echo "Creating environment configuration from template..."
    cp .env.docker.template .env.docker
    echo "⚠️  Please edit .env.docker with your actual credentials!"
fi

# Create necessary directories
mkdir -p data logs inference/outputs backups

# Build and start containers
echo "Building and starting Docker containers..."
docker compose build
docker compose up -d

echo "Initial setup complete!"
EOF

    echo -e "${GREEN}Initial setup complete!${NC}"
    echo -e "${YELLOW}⚠️  Important: SSH to the server and edit .env.docker with your credentials${NC}"
    echo "  ssh $REMOTE_USER@$REMOTE_HOST"
    echo "  cd $REMOTE_DIR"
    echo "  nano .env.docker"
    echo "  docker compose restart"

else
    # Regular deployment
    echo -e "${YELLOW}Step 1: Pushing to Git repository...${NC}"
    git add -A
    git commit -m "Deployment update $(date +%Y-%m-%d_%H:%M:%S)" || echo "No changes to commit"
    git push

    # Step 2: Deploy on remote server
    echo -e "${YELLOW}Step 2: Deploying on remote server...${NC}"
    ssh "$REMOTE_USER@$REMOTE_HOST" << 'ENDSSH'
set -e
cd ~/risk-tool

# Check if it's a git repository
if [ ! -d ".git" ]; then
    echo "Error: Remote directory is not a git repository. Run with --initial-setup first."
    exit 1
fi

# Pull latest changes
echo "Pulling latest changes..."
git pull

# Stop existing containers
echo "Stopping containers..."
docker compose down

# Rebuild and start
echo "Building and starting containers..."
docker compose build
docker compose up -d

# Show status
echo ""
echo "Deployment complete! Container status:"
docker compose ps
echo ""
echo "Recent logs:"
docker compose logs --tail=20
ENDSSH

    echo -e "${GREEN}=== Deployment Complete! ===${NC}"
fi

echo ""
echo "Useful commands:"
echo "  Monitor logs: ssh $REMOTE_USER@$REMOTE_HOST 'cd ~/risk-tool && docker compose logs -f'"
echo "  Check status: ssh $REMOTE_USER@$REMOTE_HOST 'cd ~/risk-tool && docker compose ps'"
echo "  Manual signals: ssh $REMOTE_USER@$REMOTE_HOST 'cd ~/risk-tool && docker compose exec risk-tool python send_daily_signals.py'"
