# Risk Tool Docker Deployment Guide

## Deployment Methods

### Method 1: Git-Based Deployment (Recommended)

#### Initial Setup
```bash
# 1. Set up Git repository (GitHub/GitLab)
git remote add origin https://github.com/your-username/risk-tool.git
git push -u origin main

# 2. Run initial deployment
./deploy-git.sh --initial-setup
```

#### Future Updates
```bash
# Simply run this for all future deployments
./deploy-git.sh
```

### Method 2: Direct File Transfer

#### Initial Setup
```bash
# Copy the template and edit with your values
cp .env.docker.template .env.docker
nano .env.docker

# Run the deployment script
./deploy.sh
```

### SSH Server Prerequisites
Ensure the remote server has:
- Docker and Docker Compose installed
- User `risk-tool` created with Docker permissions
- SSH key authentication configured
- Git installed (for git-based deployment)

## Daily Operations

### Redeployment After Code Changes

#### Git-Based Redeploy (Recommended)
When you make code changes and want to redeploy:

```bash
# This will commit, push, pull, and redeploy
./deploy-git.sh
```

#### Direct File Transfer (Alternative)
```bash
# This will rebuild, backup DB, and redeploy
./deploy.sh

# Skip database backup (faster)
./deploy.sh --no-backup
```

#### Manual Steps (If Needed)
```bash
# 1. Build locally
docker build -t risk-tool:latest .

# 2. Save image
docker save risk-tool:latest | gzip > risk-tool-latest.tar.gz

# 3. Transfer to server
scp risk-tool-latest.tar.gz risk-tool@195.158.30.167:~/risk-tool/

# 4. SSH to server and load
ssh risk-tool@195.158.30.167
cd ~/risk-tool
docker load < risk-tool-latest.tar.gz
docker compose down
docker compose up -d
```

### Monitoring and Maintenance

#### Check Application Status
```bash
# SSH to server
ssh risk-tool@195.158.30.167
cd ~/risk-tool

# View container status
docker compose ps

# View logs
docker compose logs -f

# View only risk-tool logs
docker compose logs -f risk-tool

# Check cron logs specifically
docker compose exec risk-tool tail -f /app/logs/cron.log
```

#### Manual Signal Generation
```bash
# Run signals manually (useful for testing)
ssh risk-tool@195.158.30.167
cd ~/risk-tool
docker compose exec risk-tool python send_daily_signals.py
```

#### Database Operations
```bash
# Backup database manually
ssh risk-tool@195.158.30.167
cd ~/risk-tool
cp data/risk_tool.db backups/risk_tool_manual_$(date +%Y%m%d).db

# Update database manually
docker compose exec risk-tool python scripts/update_database.py
```

## Common Scenarios

### 1. Change Email Recipients
```bash
# Edit local .env.docker
nano .env.docker
# Update EMAIL_RECIPIENTS

# Deploy changes
./deploy.sh --no-backup
```

### 2. Change Cron Schedule
```bash
# Edit docker/crontab
nano docker/crontab
# Change the schedule (default: 0 8 * * * = 8AM UTC)

# Redeploy
./deploy.sh
```

### 3. Debug Failed Daily Run
```bash
ssh risk-tool@195.158.30.167
cd ~/risk-tool

# Check automation logs
docker compose exec risk-tool ls -la logs/
docker compose exec risk-tool tail -100 logs/daily_automation_*.log

# Check cron logs
docker compose exec risk-tool tail -f /app/logs/cron.log

# Run automation manually with verbose output
docker compose exec risk-tool python scripts/daily_automation.py --verbose
```

### 4. Emergency Rollback
```bash
# If deployment fails, restore previous database
ssh risk-tool@195.158.30.167
cd ~/risk-tool
docker compose down
cp backups/risk_tool_[TIMESTAMP].db data/risk_tool.db
# Deploy previous version or fix issues
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker compose logs risk-tool

# Common issues:
# - Missing .env.docker file
# - Database permission issues
# - Missing directories
```

### Emails Not Sending
```bash
# Test email configuration
docker compose exec risk-tool python -c "
from inference.email_service import EmailService
service = EmailService()
print('Email configuration loaded successfully')
"

# Check logs for SMTP errors
docker compose logs risk-tool | grep -i smtp
```

### Cron Not Running
```bash
# Check if cron is running
docker compose exec risk-tool ps aux | grep cron

# Check cron logs
docker compose exec risk-tool tail -f /app/logs/cron.log

# Manually trigger cron job
docker compose exec risk-tool /home/riskuser/.local/bin/python scripts/daily_automation.py
```

## Development Workflow

1. **Make changes locally**
2. **Test with local Docker**:
   ```bash
   docker compose up --build
   ```
3. **Deploy to production**:
   ```bash
   ./deploy.sh
   ```
4. **Monitor deployment**:
   ```bash
   ssh risk-tool@195.158.30.167 "cd ~/risk-tool && docker compose logs -f"
   ```

## Security Notes

- Never commit `.env.docker` to Git (it's in .gitignore)
- Use app passwords for Gmail, not regular passwords
- Regularly backup your database
- Monitor logs for any suspicious activity
