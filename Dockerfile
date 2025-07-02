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

# Setup cron job for daily automation
COPY --chown=root:root docker/crontab /etc/cron.d/risk-tool-cron
RUN chmod 0644 /etc/cron.d/risk-tool-cron && \
    crontab -u riskuser /etc/cron.d/risk-tool-cron

# Switch to non-root user
USER riskuser

# Update PATH
ENV PATH=/home/riskuser/.local/bin:$PATH

# Create entrypoint script
COPY --chown=riskuser:riskuser docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose Streamlit port (if needed for dashboard)
EXPOSE 8501

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["cron", "-f"]
