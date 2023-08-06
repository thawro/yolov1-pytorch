import psutil
import GPUtil
from abc import abstractmethod
from src.utils.utils import merge_dicts
from typing import Any


class BaseMonitor:
    name: str = ""

    @property
    @abstractmethod
    def count(self) -> int:
        return psutil.cpu_count()

    @property
    @abstractmethod
    def temperature(self) -> float:
        raise NotImplementedError()

    @property
    @abstractmethod
    def memory_used(self) -> list[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def memory_total(self) -> list[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def memory_percent(self) -> list[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def utilization(self) -> list[int]:
        raise NotImplementedError()

    def info(self) -> dict[str, int | float | list[int] | list[float]]:
        return {
            "Temperature C": round(self.temperature, 2),
            "Total memory MB": self.memory_total,
            "Used memory MB": self.memory_used,
            "Memory utilization pct": self.memory_percent,
            f"{self.name} utilization pct": self.utilization,
        }


class CPUMonitor(BaseMonitor):
    name: str = "CPU"

    @property
    def count(self) -> int:
        return psutil.cpu_count()

    @property
    def temperature(self) -> float:
        temps = psutil.sensors_temperatures()["coretemp"]
        temps = [temp.current for temp in temps]
        return sum(temps) / len(temps)

    @property
    def memory_total(self) -> int:
        return int(psutil.virtual_memory().total >> 20)

    @property
    def memory_used(self) -> int:
        return int(psutil.virtual_memory().used >> 20)

    @property
    def memory_percent(self) -> float:
        return psutil.virtual_memory().percent

    @property
    def utilization(self) -> float | list[float]:
        return psutil.cpu_percent(percpu=False)


class GPUMonitor(BaseMonitor):
    name: str = "GPU"

    def __init__(self, device_idx: int = 0):
        self.device_idx = device_idx

    @property
    def gpu(self) -> GPUtil.GPU:
        return GPUtil.getGPUs()[self.device_idx]

    @property
    def temperature(self) -> float:
        return self.gpu.temperature

    @property
    def memory_total(self) -> int:
        return int(self.gpu.memoryTotal)

    @property
    def memory_used(self) -> int:
        return int(self.gpu.memoryUsed)

    @property
    def memory_percent(self) -> float:
        return self.gpu.memoryUtil

    @property
    def utilization(self) -> float:
        return self.gpu.load * 100


class Monitor:
    def __init__(self):
        self.cpu = CPUMonitor()
        self.gpu = GPUMonitor()

    @property
    def metrics(self) -> dict[str, Any]:
        cpu_metrics = self.cpu.info()
        gpu_metrics = self.gpu.info()
        return merge_dicts(sep="/", cpu=cpu_metrics, gpu=gpu_metrics)
