from tqdm import tqdm
import urllib.request
from src.logging.pylogger import get_pylogger
from typing import Any
import torch
from torch import nn
import torch.utils.data
from datetime import datetime

# datetime object containing current date and time

log = get_pylogger(__name__)


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


def download_file(url: str, filepath: str):
    log.info(f"Downloading {url} to {filepath}.")
    with TqdmUpTo(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split("/")[-1]
    ) as t:  # all optional kwargs
        urllib.request.urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n
    log.info("Download finished.")


def save_txt_to_file(txt: str, filename: str):
    with open(filename, "w") as file:
        file.write(txt)


def read_text_file(filename: str) -> list[str]:
    with open(filename, "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]  # Optional: Remove leading/trailing whitespace
    return lines


def train_loop(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.modules.loss._Loss,
    device: str = "cpu",
    limit_batches: int = -1,
) -> dict[str, float]:
    loop = tqdm(dataloader, leave=True, desc="Train")
    losses = []

    for x, y in loop:
        if limit_batches == 0:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        limit_batches -= 1
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss}


def merge_dicts(sep: str = "/", **dict_of_dicts) -> dict[str, Any]:
    merged_dict = {}
    for outer_name, dict in dict_of_dicts.items():
        for inner_name, value in dict.items():
            merged_dict[f"{outer_name}{sep}{inner_name}"] = value
    return merged_dict


def display_metrics(prefix: str, metrics: dict[str, Any]):
    metrics_msg = ",   ".join([f"{name} = {value:.2f}" for name, value in metrics.items()])
    log.info(prefix + metrics_msg)


def get_current_date_and_time() -> str:
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    return dt_string
