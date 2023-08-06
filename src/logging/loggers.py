from pathlib import Path
import yaml
from typing import Any
from src.logging.pylogger import get_pylogger
from src.logging.monitoring import Monitor
import mlflow
from collections import defaultdict

log = get_pylogger(__name__)


class Results:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.steps: dict[str, list[int]] = defaultdict(lambda: [], {})
        self.metrics: dict[str, list[float]] = defaultdict(lambda: [], {})
        self.params: dict[str, Any] = {}

    def update_metrics(self, metrics: dict[str, float], step: int | None = None):
        for name, value in metrics.items():
            self.metrics[name].append(value)
            if step is not None:
                self.steps[name].append(step)

    def update_params(self, params: dict[str, Any]):
        self.params.update(params)

    def get_metrics(self) -> dict[str, dict[str, list[int | float]]]:
        metrics = {name: {} for name in self.metrics}
        for name in self.metrics:
            metrics[name]["value"] = self.metrics[name]
            if name in self.steps:
                metrics[name]["step"] = self.steps[name]
        return metrics


class BaseLogger:
    def __init__(self, log_path: str | Path = "results", config: dict[str, Any] = {}):
        self.log_path = Path(log_path) if isinstance(log_path, str) else log_path
        self.ckpt_dir = self.log_path / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = str(self.ckpt_dir / "best.pt")
        self.last_ckpt_path = str(self.ckpt_dir / "last.pt")
        self.monitor = Monitor()
        self.results = Results(config=config)
        self.log_dict(config, "config.yaml")

    def log(self, key: str, value: float, step: int | None = None):
        self.results.update_metrics({key: value}, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        self.results.update_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]):
        self.results.update_params(params)

    def log_monitoring(self, step: int | None = None):
        monitoring_metrics = self.monitor.metrics
        self.log_metrics(monitoring_metrics, step=step)

    def log_dict(self, config: dict[str, Any], filename: str = "config.yaml"):
        path = str(self.log_path / filename)
        with open(path, "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        pass

    def log_checkpoints(self):
        path = str(self.log_path / "checkpoints")
        self.log_artifact(path)

    def log_model(self, model, dummy_input):
        model_dir_path = self.log_path / "models"
        model_dir_path.mkdir(parents=True, exist_ok=True)
        try:
            model.export_to_onnx(dummy_input, filepath=str(model_dir_path / "model.onnx"))
        except Exception as e:  # onnx may not support some layers
            log.error(e)
        try:
            # Save model modules and summary
            model.export_to_txt(filepath=str(model_dir_path / "model_modules.txt"))
            model.export_summary_to_txt(
                dummy_input, filepath=str(model_dir_path / "model_summary.txt")
            )
        except RuntimeError as e:
            log.error(e)

        self.log_artifact(str(model_dir_path), "models")

    def log_experiment(self):
        log.info("Logging experiment")
        metrics = self.results.get_metrics()
        self.log_dict(metrics, "metrics.yaml")
        self.log_checkpoints()
        self.finalize()

    def finalize(self):
        log.info(f"Experiment finished. Closing {self.__class__.__name__}")


class TerminalLogger(BaseLogger):
    def log(self, key: str, value: float, step: int | None = None):
        super().log(key, value, step)
        log.info(f"Step {step}, {key}: {value}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        super().log_metrics(metrics, step)
        log.info(f"Step {step}, Metrics: {metrics}")

    def log_params(self, params: dict[str, Any]):
        super().log_params(params)
        log.info(f"Params: {params}")


class MLFlowLogger(BaseLogger):
    def __init__(
        self,
        log_path: str | Path,
        config: dict[str, Any],
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        run_id: str | None = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        if tracking_uri is None:
            tracking_uri = "http://0.0.0.0:5000"
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            run_name=run_name,
            description=None,
        )
        super().__init__(log_path=log_path, config=config)

    @property
    def run(self) -> mlflow.ActiveRun | None:
        return mlflow.active_run()

    @property
    def run_id(self):
        return self.run.info.run_id

    def log(self, key: str, value: float, step: int | None = None):
        super().log(key, value, step)
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        super().log_metrics(metrics, step)
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]):
        super().log_params(params)
        mlflow.log_params(params)

    def log_dict(self, config: dict[str, Any], filename: str = "config.yaml"):
        super().log_dict(config, filename)
        mlflow.log_dict(config, filename)

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        mlflow.log_artifact(local_path, artifact_path)

    def finalize(self):
        super().finalize()
        mlflow.end_run()
