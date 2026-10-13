"""Expose the latest RAGAS faithfulness from MLflow as a Prometheus gauge."""
import logging
import os
import time

from mlflow.tracking import MlflowClient
from prometheus_client import Gauge, start_http_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

g = Gauge("ragas_faithfulness", "Latest mean RAGAS faithfulness from MLflow")


def latest_faithfulness() -> float | None:
    try:
        client = MlflowClient(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        )
        exp = client.get_experiment_by_name("ragas_evaluation")
        if exp is None:
            return None
        runs = client.search_runs(
            exp.experiment_id, order_by=["start_time DESC"], max_results=1
        )
        return runs[0].data.metrics.get("mean_faithfulness") if runs else None
    except Exception as e:
        logger.warning(f"Waiting for MLflow database/experiment: {e}")
        return None


if __name__ == "__main__":
    start_http_server(9100)
    logger.info("Faithfulness exporter running on port :9100")
    while True:
        v = latest_faithfulness()
        if v is not None:
            g.set(v)
        time.sleep(30)