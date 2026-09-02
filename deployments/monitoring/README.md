# Monitoring

Native H100 host:

```bash
docker compose -f deployments/monitoring/docker-compose.yaml up -d
```

Kubernetes:

```bash
enginebench deploy monitoring --execute
```

The native stack scrapes the active engine at port 8000 and NVIDIA DCGM at
port 9400. The React dashboard is for immutable benchmark comparisons; Grafana
is for live GPU/server telemetry during a run.
