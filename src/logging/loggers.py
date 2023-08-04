from mlflow import MlflowClient
from mlflow.entities import Param, Metric
from abc import abstractmethod
from pathlib import Path
import yaml
from typing import Any
from src.logging.pylogger import get_pylogger
from src.logging.monitoring import CPUMonitor, GPUMonitor
from src.utils.utils import merge_dicts

log = get_pylogger(__name__)


class BaseLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path) if isinstance(log_path, str) else log_path
        self.ckpt_dir = self.log_path / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = str(self.ckpt_dir / "best.pt")
        self.last_ckpt_path = str(self.ckpt_dir / "last.pt")
        self.cpu = CPUMonitor()
        self.gpu = GPUMonitor()

    def log_monitoring(self, step: int | None = None, timestamp: int | None = None):
        cpu_metrics = self.cpu.info()
        gpu_metrics = self.gpu.info()
        monitoring_metrics = merge_dicts(sep="/", cpu=cpu_metrics, gpu=gpu_metrics)
        self.log_metrics(monitoring_metrics, step=step, timestamp=timestamp)

    def log_params(self, params_dict: dict[str, Any]):
        pass

    def log(self, key: str, value: float, step: int | None = None, timestamp: int | None = None):
        pass

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None, timestamp: int | None = None
    ):
        pass

    def log_artifact(self, local_path: str, artifact_path: str):
        pass

    def log_config(self, config: dict):
        path = str(self.log_path / "config.yaml")
        with open(path, "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)
        self.log_artifact(path, "config.yaml")

    def log_checkpoints(self):
        path = str(self.log_path / "checkpoints")
        self.log_artifact(path, "checkpoints")

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


class TerminalLogger(BaseLogger):
    def log_config(self, config: dict[str, Any]):
        log.info(f"Config: {config}")

    def log_params(self, params_dict: dict[str, Any]):
        log.info(f"Params: {params_dict}")

    def log(self, key: str, value: float, step: int | None = None, timestamp: int | None = None):
        log.info(f"Step {step}, {key}: {value}")

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None, timestamp: int | None = None
    ):
        log.info(f"Step {step}, Metrics: {metrics}")


class MLFlowLogger(BaseLogger):
    def __init__(
        self,
        log_path: str,
        tracking_uri: str,
        experiment_name: str,
        run_id: str | None = None,
        run_name: str | None = None,
    ):
        super().__init__(log_path=log_path)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._run_id = run_id
        self.run_name = run_name
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment = self._get_experiment()
        self.run = self._get_run()

    def _get_experiment(self):
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(self.experiment_name)
            experiment = self.client.get_experiment(experiment_id)

        return experiment

    def _get_run(self):
        if self._run_id is not None:
            return self.client.get_run(self._run_id)
        return self.client.create_run(self.experiment.experiment_id, run_name=self.run_name)

    @property
    def run_id(self):
        return self.run.info.run_id

    def log_config(self, config: dict[str, Any]):
        self.client.log_dict(self.run_id, config, "config.yaml")

    def log_params(self, params_dict: dict[str, Any]):
        params_seq = [Param(key, str(value)) for key, value in params_dict.items()]
        self.client.log_batch(run_id=self.run_id, params=params_seq)

    def log(self, key: str, value: float, step: int | None = None, timestamp: int | None = None):
        self.client.log_metric(self.run_id, key, value=value, step=step, timestamp=timestamp)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None, timestamp: int | None = None
    ):
        metrics_seq = [Metric(key, value, timestamp, step) for key, value in metrics.items()]
        self.client.log_batch(run_id=self.run_id, metrics=metrics_seq)
        # for param, value in metrics.items():
        #     self.log(param, value, step=step, timestamp=timestamp)

    def log_artifact(self, local_path: str, artifact_path: str):
        self.client.log_artifact(self.run_id, local_path, artifact_path)
