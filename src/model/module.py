from torch import nn
import torch
from src.data.dataset import DataModule
from .classifier import YOLOv1Classifier
from .detector import YOLOv1Detector
from src.utils.model import load_checkpoint, save_checkpoint
from tqdm.auto import tqdm
from torchmetrics.functional import accuracy
from src.metrics import MAP
from torchinfo import summary
from src.utils.utils import merge_dicts, display_metrics
from src.logging.loggers import BaseLogger


class BaseModelModule:
    input_names: list[str] = ["image"]
    output_names: list[str] = ["output"]

    def __init__(
        self,
        model: YOLOv1Classifier | YOLOv1Detector | nn.Module,
        datamodule: DataModule,
        loss_fn: torch.nn.modules.loss._Loss,
        logger: BaseLogger,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        limit_batches: int = -1,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.datamodule = datamodule
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self._limit_batches = limit_batches
        self.device = device
        self.epoch = 0
        self.best_loss = torch.inf

    @staticmethod
    def _update_metrics(metrics: dict[str, list[float]], batch_metrics: dict[str, float]):
        for metric_name in batch_metrics:
            if metric_name not in metrics:
                metrics[metric_name] = [batch_metrics[metric_name]]
            else:
                metrics[metric_name].append(batch_metrics[metric_name])

    @staticmethod
    def _reduce_metrics(metrics: dict[str, list[float]]) -> dict[str, float]:
        return {name: sum(values) / len(values) for name, values in metrics.items()}

    @staticmethod
    def _add_prefix(metrics: dict[str, float], prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{name}": value for name, value in metrics.items()}

    def train_step(self, batch: tuple) -> dict[str, float]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        out = self.model(x)
        loss = self.loss_fn(out, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def train_epoch(self):
        loop = tqdm(self.datamodule.train_dataloader(), leave=True, desc="Train")
        metrics = {}
        limit_batches = int(self._limit_batches)
        for batch in loop:
            if limit_batches == 0:
                break
            batch_metrics = self.train_step(batch)
            self._update_metrics(metrics, batch_metrics)
            loop.set_postfix(loss=batch_metrics["loss"])
            limit_batches -= 1
        mean_metrics = self._reduce_metrics(metrics)
        self.logger.log_metrics(self._add_prefix(mean_metrics, prefix="train/"), step=self.epoch)
        if self.scheduler is not None:
            self.scheduler.step()
        return mean_metrics

    def val_step(self, batch: tuple) -> dict[str, float]:
        raise NotImplementedError()

    def val_epoch(self) -> dict[str, float]:
        loop = tqdm(self.datamodule.val_dataloader(), desc="Val")
        metrics = {}
        limit_batches = int(self._limit_batches)
        self.model.eval()
        for batch in loop:
            if limit_batches == 0:
                break
            batch_metrics = self.val_step(batch)
            self._update_metrics(metrics, batch_metrics)
            limit_batches -= 1
        mean_metrics = self._reduce_metrics(metrics)
        self.logger.log_metrics(self._add_prefix(mean_metrics, prefix="val/"), step=self.epoch)
        self.model.train()
        return mean_metrics

    def test_step(self, batch: tuple) -> dict[str, float]:
        raise NotImplementedError()

    def test_epoch(self) -> dict[str, float]:
        raise NotImplementedError()

    def _display_metrics(
        self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ):
        metrics = merge_dicts(sep="/", train=train_metrics, val=val_metrics)
        lr = self.optimizer.param_groups[0]["lr"]
        prefix = f"Epoch: {epoch}, LR = {lr:.5f}  |  "
        display_metrics(prefix=prefix, metrics=metrics)

    def _common_loops(self):
        train_metrics = self.train_epoch()
        val_metrics = self.val_epoch()
        return train_metrics, val_metrics

    def _epoch_end(self, train_metrics: dict[str, float], val_metrics: dict[str, float]):
        self._display_metrics(self.epoch, train_metrics, val_metrics)
        self.logger.log_monitoring(step=self.epoch)
        self.logger.log("learning_rate", self.optimizer.param_groups[0]["lr"], step=self.epoch)
        self.epoch += 1

    def fit(self, epochs: int, ckpt_path: str | None = None):
        try:
            self.logger.log_config()
            if ckpt_path is not None:
                self.load_checkpoint(ckpt_path)

            for epoch in range(epochs):
                train_metrics, val_metrics = self._common_loops()
                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
                    self.save_checkpoint(self.logger.best_ckpt_path)
                self.save_checkpoint(self.logger.last_ckpt_path)
                self._epoch_end(train_metrics, val_metrics)
            self.logger.log_experiment()
        except KeyboardInterrupt as e:
            self.logger.log_experiment()
            raise e

    def export_to_onnx(self, dummy_input: dict[str, torch.Tensor], filepath: str):
        torch.onnx.export(
            self.model,
            dummy_input,
            filepath,
            verbose=False,
            input_names=self.input_names,
            output_names=self.output_names,
            export_params=True,
        )

    def export_to_txt(self, filepath: str, save: bool = False) -> str:
        modules_txt = str(self.model)
        if save:
            with open(filepath, "w") as text_file:
                text_file.write(modules_txt)
        return modules_txt

    def export_summary_to_txt(
        self, dummy_input: dict[str, torch.Tensor], filepath: str, save: bool = False
    ):
        model_summary = summary(
            self.model,
            input_data=dummy_input,
            depth=10,
            col_names=["input_size", "output_size", "num_params", "kernel_size"],
            verbose=0,
            device=self.device,
        )
        if save:
            with open(filepath, "w") as text_file:
                text_file.write(str(model_summary))
        return model_summary

    def load_checkpoint(self, ckpt_path: str):
        ckpt = load_checkpoint(ckpt_path, self.model, self.optimizer, self.scheduler)
        self.best_loss = ckpt["best_loss"]
        self.epoch = ckpt["epoch"] + 1

    def save_checkpoint(self, ckpt_path: str):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "best_loss": self.best_loss,
        }
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()
        save_checkpoint(ckpt, ckpt_path)


class ClassificationModelModule(BaseModelModule):
    model: YOLOv1Classifier

    def val_step(self, batch: tuple) -> dict[str, float]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            out = self.model(x)
        loss = self.loss_fn(out, y).item()
        acc = accuracy(out, y, task="multiclass", num_classes=self.model.num_classes).item()
        return {"loss": loss, "acc": acc}

    def save_checkpoint(self, ckpt_path: str):
        super().save_checkpoint(ckpt_path)
        backbone_ckpt = {"model": self.model.backbone.state_dict()}
        if "best" in ckpt_path:
            backbone_ckpt_path = ckpt_path.replace("best", "backbone_best")
        else:
            backbone_ckpt_path = ckpt_path.replace("last", "backbone_last")
        save_checkpoint(backbone_ckpt, path=backbone_ckpt_path)


class DetectionModelModule(BaseModelModule):
    model: YOLOv1Detector

    def val_step(self, batch: tuple) -> dict[str, float]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        batch_size = x.shape[0]

        all_pred_boxes = []
        all_true_boxes = []

        with torch.no_grad():
            out = self.model(x)

        loss = self.loss_fn(out, y).item()
        pred_boxes = self.model.cellboxes_to_boxes(out)
        pred_boxes = self.model.perform_nms(pred_boxes)
        true_boxes = self.model.cellboxes_to_boxes(y)

        train_idx = 0
        for idx in range(batch_size):
            for nms_box in pred_boxes[idx]:
                all_pred_boxes.append([train_idx] + nms_box)
            for box in true_boxes[idx]:
                if box[1] > self.model.obj_threshold:
                    all_true_boxes.append([train_idx] + box.tolist())
            train_idx += 1

        mean_avg_prec = MAP(
            self.model.C, all_pred_boxes, all_true_boxes, iou_threshold=self.model.iou_threshold
        )
        return {"loss": loss, "MAP": mean_avg_prec}
