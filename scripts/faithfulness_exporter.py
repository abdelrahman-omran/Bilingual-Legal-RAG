"""Expose the latest completed RAGAS faithfulness run from MLflow to Prometheus.

Run from the repository root, or configure MLFLOW_TRACKING_URI explicitly.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import NamedTuple

from mlflow.tracking import MlflowClient
from prometheus_client import Gauge, start_http_server

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ragas_evaluation")
POLL_SECONDS = max(5, int(os.getenv("FAITHFULNESS_POLL_SECONDS", "30")))

# Prefer the mean expected by the original exporter. Support `faithfulness` as a
# fallback for evaluators that log RAGAS's aggregate metric under that key.
PRIMARY_METRIC = os.getenv("MLFLOW_FAITHFULNESS_METRIC", "mean_faithfulness")
METRIC_CANDIDATES = list(dict.fromkeys([PRIMARY_METRIC, "mean_faithfulness", "faithfulness"]))

faithfulness_gauge = Gauge(
    "ragas_faithfulness",
    "Faithfulness score from the latest completed RAGAS evaluation run (0 to 1)",
)
faithfulness_alias = Gauge(
    "faithfulness",
    "Latest valid per-question RAGAS faithfulness score loaded from MLflow (0 to 1)",
)
faithfulness_available = Gauge(
    "ragas_faithfulness_available",
    "1 if a completed RAGAS faithfulness result has been loaded from MLflow, otherwise 0",
)

# Prometheus Gauge starts at 0 by default. NaN makes “not computed yet” distinct
# from a real RAGAS score of 0.0.
faithfulness_gauge.set(float("nan"))
faithfulness_alias.set(float("nan"))
faithfulness_available.set(0)


class FaithfulnessResult(NamedTuple):
    score: float
    run_id: str
    metric_name: str
    end_time: int


def latest_faithfulness() -> FaithfulnessResult | None:
    """Return the newest valid faithfulness score from a FINISHED MLflow run.

    In-progress runs and completed runs that don't contain a faithfulness metric
    are skipped, so they cannot hide the latest valid completed score.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        logger.debug("MLflow experiment %r has not been created yet", EXPERIMENT_NAME)
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1000,
    )
    # Search may return runs in start-time order. Sort by completion time so a
    # run that finished later wins even if it started earlier.
    runs.sort(key=lambda run: run.info.end_time or 0, reverse=True)

    for run in runs:
        metrics = run.data.metrics
        for metric_name in METRIC_CANDIDATES:
            value = metrics.get(metric_name)
            if value is None:
                continue
            score = float(value)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                logger.warning(
                    "Skipping invalid %s=%r in MLflow run %s",
                    metric_name,
                    value,
                    run.info.run_id,
                )
                continue
            return FaithfulnessResult(
                score=score,
                run_id=run.info.run_id,
                metric_name=metric_name,
                end_time=run.info.end_time or 0,
            )

    logger.debug(
        "No FINISHED run in experiment %r contains any of these metrics: %s",
        EXPERIMENT_NAME,
        ", ".join(METRIC_CANDIDATES),
    )
    return None


def main() -> None:
    global_last_run_id: str | None = None
    start_http_server(9100)
    logger.info(
        "Faithfulness exporter listening on :9100; tracking_uri=%s; experiment=%s; metric candidates=%s",
        TRACKING_URI,
        EXPERIMENT_NAME,
        METRIC_CANDIDATES,
    )

    while True:
        try:
            result = latest_faithfulness()
            if result is not None:
                faithfulness_gauge.set(result.score)
                faithfulness_alias.set(result.score)
                faithfulness_available.set(1)
                if result.run_id != global_last_run_id:
                    logger.info(
                        "Loaded %s=%.4f from completed MLflow run %s (experiment=%s)",
                        result.metric_name,
                        result.score,
                        result.run_id,
                        EXPERIMENT_NAME,
                    )
                    global_last_run_id = result.run_id
        except Exception:
            # Keep the last known valid score during temporary DB/MLflow errors.
            # At startup the score remains NaN and *_available remains 0.
            logger.exception("Failed to read faithfulness from MLflow")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
