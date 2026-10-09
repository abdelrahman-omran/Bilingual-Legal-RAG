import json
import os
import sys
import urllib.request

from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
exp = client.get_experiment_by_name("ragas_evaluation")
if exp is None:
    sys.exit("No 'ragas_evaluation' experiment found — run evaluate_ragas.py first.")

runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
faith = runs[0].data.metrics.get("mean_faithfulness")

if faith is None:
    sys.exit("Latest run has no faithfulness metric.")
if faith < 0.80:
    msg = f"🚨 RAGAS faithfulness {faith:.3f} < 0.80 (run {runs[0].info.run_id})"
    webhook = os.getenv("ALERT_WEBHOOK_URL")
    if webhook:
        urllib.request.urlopen(urllib.request.Request(
            webhook, data=json.dumps({"text": msg}).encode(),
            headers={"Content-Type": "application/json"}))
    print(msg)
    sys.exit(1)
print(f"✅ faithfulness {faith:.3f} ≥ 0.80")