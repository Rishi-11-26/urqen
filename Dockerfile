# ==============================================================================
# URQEN — Production Multi-Stage Container for Render / Cloud Run
# Stage 1: Build React Frontend
# Stage 2: Python ML Engine & FastAPI Production Server
# ==============================================================================

# --- Stage 1: Frontend Build ---
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python Backend Runtime ---
FROM python:3.11-slim AS runner

# System dependencies for scientific packages & compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code and URQEN ML engine
COPY backend/ ./backend/
COPY previa/ ./previa/

# Copy built frontend assets from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Ensure storage directories exist
RUN mkdir -p storage/datasets storage/artifacts storage/metadata

# Render passes PORT dynamically
ENV PORT=8000
EXPOSE 8000

CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
