FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cron \
    sqlite3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy docker configuration
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/crontab /etc/cron.d/risk-tool

# Set permissions
RUN chmod +x /entrypoint.sh
RUN chmod 0644 /etc/cron.d/risk-tool

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/inference/outputs

# Apply cron job
RUN crontab /etc/cron.d/risk-tool

# Expose port for potential web interface
EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
CMD ["cron"]
