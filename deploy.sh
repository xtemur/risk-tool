#!/bin/bash
# Risk Tool Remote Deployment Script
# Usage: ./deploy.sh [--build-only] [--no-backup]

set -e

# Configuration
REMOTE_USER="risk-tool"
REMOTE_HOST="195.158.30.167"
REMOTE_DIR="/home/risk-tool/risk-tool"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
BUILD_ONLY=false
NO_BACKUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --no-backup)
            NO_BACKUP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--build-only] [--no-backup]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== Risk Tool Deployment Script ===${NC}"
echo "Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
echo ""

# Step 1: Build Docker image locally
echo -e "${YELLOW}Step 1: Building Docker image...${NC}"
docker build -t risk-tool:latest .

if [ "$BUILD_ONLY" = true ]; then
    echo -e "${GREEN}Build complete. Exiting (--build-only flag).${NC}"
    exit 0
fi

# Step 2: Save Docker image
echo -e "${YELLOW}Step 2: Saving Docker image...${NC}"
docker save risk-tool:latest | gzip > risk-tool-latest.tar.gz

# Step 3: Check remote connection
echo -e "${YELLOW}Step 3: Checking remote connection...${NC}"
if ! ssh -o ConnectTimeout=10 "$REMOTE_USER@$REMOTE_HOST" "echo 'Connection successful'"; then
    echo -e "${RED}Failed to connect to remote server${NC}"
    exit 1
fi

# Step 4: Create remote directory if needed
echo -e "${YELLOW}Step 4: Setting up remote directory...${NC}"
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

# Step 5: Backup database (if not disabled)
if [ "$NO_BACKUP" = false ]; then
    echo -e "${YELLOW}Step 5: Backing up remote database...${NC}"
    ssh "$REMOTE_USER@$REMOTE_HOST" "
        if [ -f $REMOTE_DIR/data/risk_tool.db ]; then
            mkdir -p $REMOTE_DIR/backups
            cp $REMOTE_DIR/data/risk_tool.db $REMOTE_DIR/backups/risk_tool_$(date +%Y%m%d_%H%M%S).db
            echo 'Database backed up successfully'
        else
            echo 'No existing database found, skipping backup'
        fi
    "
else
    echo -e "${YELLOW}Step 5: Skipping database backup (--no-backup flag)${NC}"
fi

# Step 6: Transfer files
echo -e "${YELLOW}Step 6: Transferring files to remote...${NC}"
echo "Uploading Docker image..."
scp risk-tool-latest.tar.gz "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

echo "Uploading configuration files..."
scp docker-compose.yml "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
scp -r docker "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

# Check if .env.docker exists
if [ ! -f ".env.docker" ]; then
    echo -e "${RED}Warning: .env.docker not found!${NC}"
    echo "Please create .env.docker from .env.docker.template before deploying"
    exit 1
fi
scp .env.docker "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

# Transfer necessary directories
echo "Creating remote directories..."
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR/{configs,data,logs,inference/outputs}"

echo "Uploading configs..."
scp -r configs "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

# Step 7: Load Docker image on remote
echo -e "${YELLOW}Step 7: Loading Docker image on remote...${NC}"
ssh "$REMOTE_USER@$REMOTE_HOST" "
    cd $REMOTE_DIR
    docker load < risk-tool-latest.tar.gz
    rm risk-tool-latest.tar.gz
"

# Step 8: Stop existing containers
echo -e "${YELLOW}Step 8: Stopping existing containers...${NC}"
ssh "$REMOTE_USER@$REMOTE_HOST" "
    cd $REMOTE_DIR
    docker compose down || true
"

# Step 9: Start new containers
echo -e "${YELLOW}Step 9: Starting new containers...${NC}"
ssh "$REMOTE_USER@$REMOTE_HOST" "
    cd $REMOTE_DIR
    docker compose up -d
"

# Step 10: Verify deployment
echo -e "${YELLOW}Step 10: Verifying deployment...${NC}"
sleep 5
ssh "$REMOTE_USER@$REMOTE_HOST" "
    cd $REMOTE_DIR
    echo 'Container status:'
    docker compose ps
    echo ''
    echo 'Recent logs:'
    docker compose logs --tail=20
"

# Cleanup local files
rm -f risk-tool-latest.tar.gz

echo -e "${GREEN}=== Deployment Complete! ===${NC}"
echo ""
echo "To check logs on remote server:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST"
echo "  cd $REMOTE_DIR"
echo "  docker compose logs -f"
echo ""
echo "To run manual signal generation:"
echo "  docker compose exec risk-tool python send_daily_signals.py"
