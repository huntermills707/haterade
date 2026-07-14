# Monitoring dashboards

Grafana dashboards for observing the MLOps pipeline.

## M4 — CPU Canary

`dashboards/m4-cpu-canary.json` is a Grafana dashboard for milestone M4:

- Rollout phase and AnalysisRun phase
- Rollout replicas (available / desired / updated-canary)
- Inference request rate split by model version (stable v1 vs canary v2)
- Canary success rate
- Canary queue latency
- Predictor CPU utilization by pod

### Deploy

```bash
kubectl apply -f monitoring/dashboards/k8s-configmap-m4.yaml
```

### Argo Rollouts metrics

Argo Rollouts controller metrics are not exposed by the Helm chart by default.
`monitoring/argo-rollouts-metrics.yaml` adds a Service + ServiceMonitor so
Prometheus scrapes the controller's `/metrics` endpoint on port 8090. The M4
dashboard depends on these metrics.

```bash
kubectl apply -f monitoring/argo-rollouts-metrics.yaml
```

## M3 — CPU Autoscaling

`dashboards/m3-cpu-autoscaling.json` is a Grafana dashboard for milestone M3:

- Inference request rate
- Predictor replica count
- KEDA trigger signal (Triton queue duration)
- Average inference latency
- Predictor CPU utilization

### Deploy

```bash
kubectl apply -f monitoring/dashboards/k8s-configmap.yaml
```

The ConfigMap is labeled `grafana_dashboard: "1"` so the kube-prometheus-stack
Grafana sidecar should import it automatically. If it does not appear, check
that the sidecar is watching the `observability` namespace and the correct
label.

## Istio ingress gateway metrics

`istio-ingress-metrics.yaml` adds a headless Service + ServiceMonitor that
scrapes Envoy's `/stats/prometheus` endpoint on port 15090. This is optional
for M3 but enables gateway-level observability and is a prerequisite for any
future KEDA trigger based on ingress request rate (e.g. `istio_requests_total`).

```bash
kubectl apply -f monitoring/istio-ingress-metrics.yaml
```

### Access Grafana

```bash
kubectl -n observability port-forward svc/kube-prometheus-stack-grafana 3000:80
```

Default credentials are usually `admin` / `prom-operator`. Verify with:

```bash
kubectl -n observability get secret kube-prometheus-stack-grafana -o jsonpath='{.data.admin-password}' | base64 -d; echo
```

### Import manually

If the sidecar is not enabled, import `dashboards/m3-cpu-autoscaling.json`
directly through the Grafana UI (`+` → Import).

## Dashboard development

Edit `dashboards/m3-cpu-autoscaling.json`, then regenerate the ConfigMap:

```bash
kubectl create configmap grafana-dashboard-m3-cpu-autoscaling \
  --from-file=m3-cpu-autoscaling.json=monitoring/dashboards/m3-cpu-autoscaling.json \
  -n observability --dry-run=client -o yaml | \
  kubectl label --local -f - grafana_dashboard=1 -o yaml > monitoring/dashboards/k8s-configmap.yaml
```

Edit `dashboards/m4-cpu-canary.json`, then regenerate the ConfigMap:

```bash
kubectl create configmap grafana-dashboard-m4-cpu-canary \
  --from-file=m4-cpu-canary.json=monitoring/dashboards/m4-cpu-canary.json \
  -n observability --dry-run=client -o yaml | \
  kubectl label --local -f - grafana_dashboard=1 -o yaml > monitoring/dashboards/k8s-configmap-m4.yaml
```
