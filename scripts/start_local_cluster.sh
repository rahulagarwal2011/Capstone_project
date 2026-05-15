#!/bin/bash
# Start local development cluster (Ray + Redis via Docker Compose).
# Usage: bash scripts/start_local_cluster.sh

set -euo pipefail

echo "=== Starting Reason-Reduce Local Cluster ==="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Install Docker Desktop first."
    exit 1
fi

# Start services
docker compose up -d

echo ""
echo "Services started:"
echo "  Ray Dashboard:  http://localhost:8265"
echo "  Redis:          localhost:6380"
echo "  Prometheus:     http://localhost:9090"
echo "  Grafana:        http://localhost:3000 (admin/admin)"
echo ""
echo "Run 'make smoke' to verify the pipeline."
