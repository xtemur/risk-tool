# Multi-stage Dockerfile for Risk Tool
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    cron \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone (you can change this to your timezone)
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create non-root user
RUN useradd -m -s /bin/bash riskuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/riskuser/.local

# Copy application code
COPY --chown=riskuser:riskuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/inference/outputs && \
    chown -R riskuser:riskuser /app

# Setup cron job for daily automation (run as riskuser)
RUN echo "0 8 * * * riskuser cd /app && /home/riskuser/.local/bin/python scripts/daily_automation.py --email >> /app/logs/cron.log 2>&1" > /etc/cron.d/risk-tool-cron && \
    chmod 0644 /etc/cron.d/risk-tool-cron

# Create directories with correct permissions before switching user
RUN mkdir -p /app/logs /app/data /app/inference/outputs && \
    chown -R riskuser:riskuser /app/logs /app/data /app/inference/outputs && \
    chmod 775 /app/logs /app/data /app/inference/outputs

# Update PATH for all users
ENV PATH=/home/riskuser/.local/bin:$PATH

# Create entrypoint script (as root)
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Fix permissions if needed\n\
chown -R riskuser:riskuser /app/logs /app/data /app/inference/outputs\n\
\n\
# Test database connection as riskuser\n\
echo "Testing database connection..."\n\
su - riskuser -c "cd /app && python -c \"import sqlite3; import os; os.makedirs('\''/app/data'\'', exist_ok=True); conn = sqlite3.connect('\''/app/data/risk_tool.db'\''); print('\''Database connection successful'\''); conn.close()\""\n\
\n\
# If running cron (default), start cron daemon as root\n\
if [ "$1" = "cron" ]; then\n\
    echo "Starting cron daemon for daily automation..."\n\
    # Initialize cron log\n\
    touch /app/logs/cron.log\n\
    chown riskuser:riskuser /app/logs/cron.log\n\
    # Start cron in foreground\n\
    exec cron -f\n\
else\n\
    # Otherwise, execute the command\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Expose Streamlit port (if needed for dashboard)
EXPOSE 8501

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["cron", "-f"]
