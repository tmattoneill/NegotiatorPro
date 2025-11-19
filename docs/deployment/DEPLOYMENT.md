# NegotiatorPro - Docker Deployment Guide

This guide provides step-by-step instructions for deploying NegotiatorPro using Docker and Docker Compose on Ubuntu systems.

**🚀 Architecture**: This deployment uses React frontend + FastAPI backend architecture.
- **Frontend**: React app (port 5173 in dev, served via Node or reverse proxy in production)
- **Backend**: FastAPI API server (port 8000)

**Note**: Some legacy references to port 7860 (Gradio) may remain in this document and will be updated. Use port 8000 for backend and 5173 for frontend.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Management](#management)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Maintenance](#maintenance)

## Prerequisites

### System Requirements

- Ubuntu 20.04 LTS or later
- 2GB RAM minimum (4GB recommended)
- 2 CPU cores (recommended)
- 10GB free disk space
- Internet connection for API calls

### Required Software

1. **Docker Engine** (20.10+)
2. **Docker Compose** (2.0+)

## Installation

### Step 1: Install Docker

If Docker is not already installed, install it using the official Docker installation script:

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

### Step 2: Configure Docker for Non-Root User (Optional)

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Log out and log back in for changes to take effect
# Or run: newgrp docker
```

### Step 3: Clone the Repository

```bash
# Clone the repository
git clone <repository-url>
cd NegotiatorPro

# Or if you already have the code, navigate to the directory
cd /path/to/NegotiatorPro
```

## Configuration

### Step 1: Set Up Environment Variables

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit the `.env` file and add your OpenAI API key:

```bash
nano .env
```

Update the following:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_actual_api_key_here

# Optional: Server Configuration
GRADIO_SERVER_PORT=7860
```

**Important**: Never commit the `.env` file to version control!

### Step 2: Prepare Source Documents (Optional)

If you have negotiation books or reference materials:

```bash
# Create sources directory if it doesn't exist
mkdir -p sources

# Copy your PDF, DOCX, TXT, or DOC files
cp /path/to/your/negotiation_books/*.pdf sources/
```

**Note**: The application includes a pre-built vectorstore, so this step is optional unless you want to add additional documents.

### Step 3: Configure Docker Compose (Optional)

The default `docker-compose.yml` should work for most deployments. However, you can adjust:

- **Port mapping**: Change `7860:7860` to `<your-port>:7860`
- **Resource limits**: Adjust CPU and memory limits based on your server capacity
- **Restart policy**: Change `unless-stopped` to `always` or `no` as needed

Example port change:

```yaml
ports:
  - "8080:7860"  # Access on port 8080 instead of 7860
```

## Deployment

### Quick Start (Development)

For a quick deployment:

```bash
# Build and start the container
docker compose up -d

# View logs
docker compose logs -f
```

### Production Deployment

For production, follow these steps:

#### 1. Build the Image

```bash
# Build the Docker image
docker compose build

# Verify the image was created
docker images | grep negotiator-pro
```

#### 2. Start the Service

```bash
# Start the container in detached mode
docker compose up -d

# Verify the container is running
docker compose ps
```

#### 3. Check Logs

```bash
# View real-time logs
docker compose logs -f

# View last 100 lines
docker compose logs --tail=100

# View logs for specific service
docker compose logs negotiator-pro
```

#### 4. Verify Deployment

1. Check container health:
```bash
docker compose ps
```

2. Access the application:
   - Open your browser and navigate to `http://your-server-ip:7860`
   - Or use localhost: `http://localhost:7860`

3. Test the API:
```bash
curl http://localhost:7860
```

## Management

### Starting and Stopping

```bash
# Start the service
docker compose start

# Stop the service
docker compose stop

# Restart the service
docker compose restart

# Stop and remove containers
docker compose down

# Stop and remove containers, networks, and volumes
docker compose down -v
```

### Viewing Logs

```bash
# View logs
docker compose logs

# Follow logs in real-time
docker compose logs -f

# View logs with timestamps
docker compose logs -t

# View last N lines
docker compose logs --tail=50
```

### Accessing the Container

```bash
# Execute commands in the running container
docker compose exec negotiator-pro bash

# View running processes
docker compose exec negotiator-pro ps aux

# Check Python packages
docker compose exec negotiator-pro pip list
```

### Updating the Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose up -d --build

# Or step by step:
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Managing Documents

#### Upload Documents via Web Interface

1. Access the application at `http://your-server-ip:7860`
2. Navigate to the "Admin Panel" tab
3. Log in with your admin password (default: `admin123`)
4. Go to the "Documents" tab
5. Upload PDF, TXT, DOCX, or DOC files
6. Click "Regenerate Vector Database" after uploading

#### Upload Documents via Command Line

```bash
# Copy documents to the sources directory
docker compose exec negotiator-pro cp /path/to/document.pdf /app/sources/

# Or from host machine
cp your-document.pdf sources/
```

### Regenerating Vectorstore

If you add new documents:

1. Via Web Interface: Admin Panel → Documents → Regenerate Vector Database
2. Via Container:
```bash
docker compose restart negotiator-pro
```

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs for errors
docker compose logs

# Verify .env file exists and has API key
cat .env

# Check if port is already in use
sudo netstat -tulpn | grep 7860
sudo lsof -i :7860
```

#### Permission Errors

```bash
# Fix directory permissions
sudo chown -R $USER:$USER vectorstore/ uploads/ sources/

# Or if running as root
sudo chown -R 1000:1000 vectorstore/ uploads/ sources/
```

#### Out of Memory

```bash
# Check container resource usage
docker stats

# Adjust memory limits in docker-compose.yml
# Under deploy.resources.limits.memory: increase from 2G to 4G
```

#### API Key Issues

```bash
# Verify environment variable is set
docker compose exec negotiator-pro env | grep OPENAI

# Update .env file and restart
docker compose restart
```

#### Vectorstore Errors

```bash
# Remove and regenerate vectorstore
rm -rf vectorstore/
docker compose restart negotiator-pro
```

### Checking Health

```bash
# View container health status
docker compose ps

# Check health endpoint
curl http://localhost:7860/

# View detailed container info
docker inspect negotiator-pro
```

### Log Analysis

```bash
# Search logs for errors
docker compose logs | grep -i error

# Search logs for specific text
docker compose logs | grep "Vector store"

# Save logs to file
docker compose logs > app-logs.txt
```

## Security

### Security Best Practices

1. **Change Default Admin Password**
   - Log into Admin Panel immediately after deployment
   - Navigate to "Admin Settings" → "Change Admin Password"
   - Use a strong password (12+ characters, mixed case, numbers, symbols)

2. **Protect Environment Variables**
   ```bash
   # Ensure .env is not readable by others
   chmod 600 .env

   # Verify
   ls -la .env
   ```

3. **Use Reverse Proxy**

   For production, use Nginx or Apache as a reverse proxy:

   ```nginx
   # Nginx example
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:7860;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

4. **Enable HTTPS**
   ```bash
   # Install certbot for Let's Encrypt
   sudo apt-get install certbot python3-certbot-nginx

   # Get certificate
   sudo certbot --nginx -d your-domain.com
   ```

5. **Firewall Configuration**
   ```bash
   # Allow SSH
   sudo ufw allow 22/tcp

   # Allow HTTP/HTTPS
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp

   # If not using reverse proxy, allow Gradio port
   sudo ufw allow 7860/tcp

   # Enable firewall
   sudo ufw enable
   ```

6. **Regular Updates**
   ```bash
   # Update system packages
   sudo apt-get update && sudo apt-get upgrade -y

   # Update Docker images
   docker compose pull
   docker compose up -d
   ```

### API Key Protection

- Never commit `.env` to version control
- Use Docker secrets for production (see Docker Swarm documentation)
- Rotate API keys regularly
- Monitor API usage in OpenAI dashboard

## Maintenance

### Backup

#### Backup Important Data

```bash
# Create backup directory
mkdir -p backups

# Backup vectorstore
tar -czf backups/vectorstore-$(date +%Y%m%d).tar.gz vectorstore/

# Backup sources
tar -czf backups/sources-$(date +%Y%m%d).tar.gz sources/

# Backup configuration
tar -czf backups/config-$(date +%Y%m%d).tar.gz \
    admin_config.json admin_sessions.json usage_stats.json \
    embedding_config.json prompt_config.json .env
```

#### Automated Backup Script

```bash
#!/bin/bash
# save as backup.sh
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
tar -czf "$BACKUP_DIR/negotiator-pro-$DATE.tar.gz" \
    vectorstore/ sources/ uploads/ \
    admin_config.json admin_sessions.json usage_stats.json \
    embedding_config.json prompt_config.json

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "negotiator-pro-*.tar.gz" -mtime +7 -delete

echo "Backup completed: negotiator-pro-$DATE.tar.gz"
```

#### Restore from Backup

```bash
# Stop the container
docker compose down

# Restore files
tar -xzf backups/negotiator-pro-YYYYMMDD.tar.gz

# Start the container
docker compose up -d
```

### Monitoring

#### Set Up Log Rotation

Docker automatically rotates logs, but you can adjust settings in `docker-compose.yml`:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"  # Maximum size per log file
    max-file: "3"    # Number of log files to keep
```

#### Monitor Resource Usage

```bash
# Real-time stats
docker stats negotiator-pro

# CPU and memory usage
docker compose exec negotiator-pro top
```

#### Set Up Alerts (Optional)

Use tools like:
- **Prometheus** + **Grafana** for metrics
- **Uptime Kuma** for uptime monitoring
- **Portainer** for Docker management UI

### Regular Maintenance Tasks

1. **Weekly**:
   - Review logs for errors
   - Check disk space: `df -h`
   - Verify backups are running

2. **Monthly**:
   - Update system packages: `sudo apt-get update && sudo apt-get upgrade`
   - Review and rotate admin sessions
   - Check API usage and costs

3. **Quarterly**:
   - Update Docker images
   - Review and update dependencies
   - Audit security settings

### Cleanup

```bash
# Remove unused Docker resources
docker system prune -a

# Remove old logs
find /var/lib/docker/containers -name "*.log" -mtime +30 -delete

# Clean up old backups
find backups/ -name "*.tar.gz" -mtime +30 -delete
```

## Advanced Configuration

### Running Behind a Reverse Proxy

Example Nginx configuration:

```nginx
upstream negotiator-pro {
    server localhost:7860;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://negotiator-pro;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Gradio specific settings
        proxy_buffering off;
        proxy_read_timeout 86400;
    }
}
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Server bind address |
| `GRADIO_SERVER_PORT` | `7860` | Server port |

## Support

For issues and questions:

1. Check the [main README](README.md) for project documentation
2. Review logs: `docker compose logs`
3. Check the [CLAUDE.md](CLAUDE.md) file for architecture details
4. Open an issue on the project repository

## License

See [LICENSE](LICENSE) file for details.
