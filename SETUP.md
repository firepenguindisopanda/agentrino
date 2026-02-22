# Agentrino Backend Setup Guide

## Environment Variables

Create a `.env` file in the `agentrino/` directory:

```bash
NVIDIA_API_KEY=your_nvidia_api_key_here
MONGO_URI=mongodb+srv://fullstack:password@cluster0.xxx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
MONGO_DB_NAME=agent_chatter
UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=your_upstash_token
RECENT_MESSAGES_LIMIT=30
RECENT_MESSAGES_TTL=3600
# Comma-separated list of allowed CORS origins
CORS_ORIGINS=https://your-frontend.vercel.app,http://localhost:3000
```

`uv pip compile pyproject.toml -o requirements.txt`

## Running Locally

### Option 1: With Docker (no longer needed - using Atlas + Upstash)

```bash
cd agentrino
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Option 2: Without Docker

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

cd agentrino
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## Running as a systemd Service

Create the service file:

```bash
sudo nano /etc/systemd/system/agentrino.service
```

Contents:

```ini
[Unit]
Description=Agentrino API
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/agentrino
ExecStart=/home/pi/.local/bin/uv run uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable agentrino
sudo systemctl start agentrino

# Check status
sudo systemctl status agentrino
```

---

# Deployment Guide

## Backend: Deploy to Render

1. **Push your code to GitHub**

2. **Create a new Web Service on Render**
   - Go to [render.com](https://render.com) > New > Web Service
   - Connect your GitHub repo (select the `agentrino` folder/repository)
   - Settings:
     - Runtime: Python
     - Build Command: `uv sync`
     - Start Command: `uv run uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Add Environment Variables** in Render dashboard:
   - `NVIDIA_API_KEY` = your key
   - `MONGO_URI` = your MongoDB Atlas URI
   - `MONGO_DB_NAME` = agent_chatter
   - `UPSTASH_REDIS_REST_URL` = your Upstash URL
   - `UPSTASH_REDIS_REST_TOKEN` = your Upstash token

4. **Deploy** - Render will auto-deploy on push

## Frontend: Deploy to Vercel

1. **Push your code to GitHub**

2. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com) → Add New → Project
   - Import your `agentrino_frontend` repo
   - Framework Preset: Next.js (auto-detected)

3. **Environment Variables**:
   - `NEXT_PUBLIC_API_URL` = https://your-render-api.onrender.com

4. **Deploy** - Vercel will auto-deploy on push

---

## Git Commands

### See unstaged changes:
```bash
git diff
```

### See staged changes:
```bash
git diff --staged
```

### See all changes vs last commit:
```bash
git diff HEAD~1
```

### See recent commits:
```bash
git log --oneline -5
```

### Stage all changes:
```bash
git add -A
```

### Commit changes:
```bash
git commit -m "Your message here"
```

### Push to remote:
```bash
git push origin main
```

---

## Accessing via Tailscale (Local Development)

Since you're using Tailscale:

1. **Backend** runs on `http://localhost:8000` (internal only)
2. **Frontend** needs to be exposed via Tailscale

On your **local machine**, access the Raspberry Pi services through your Tailscale IP:

```
Frontend: http://<tailscale-ip-pi>:3000
Backend:  http://<tailscale-ip-pi>:8000
```

### Exposing Frontend via Tailscale

```bash
# On the Raspberry Pi
tailscale serve http 3000
tailscale status  # Shows the URL you can share
```

---

## Verification

Test the backend is running:

```bash
curl http://localhost:8000/agents
```

Expected: JSON response (may be empty `[]` if no agents seeded)
