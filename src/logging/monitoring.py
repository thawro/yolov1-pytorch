import psutil
import GPUtil
from abc import abstractmethod


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
            "Temperature [°C]": round(self.temperature, 2),
            "Total memory [MB]": self.memory_total,
            "Used memory [MB]": self.memory_used,
            "Memory utilization [%]": self.memory_percent,
            f"{self.name} utilization [%]": self.utilization,
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
    def utilization(self) -> list[float]:
        return psutil.cpu_percent(percpu=True)


class GPUMonitor(BaseMonitor):
    name: str = "GPU"

    def __init__(self, device_idx: int = 0):
        self.gpu: GPUtil.GPU = GPUtil.getGPUs()[device_idx]

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


cpu_monitor = CPUMonitor()
gpu_monitor = GPUMonitor()

print(cpu_monitor.info())
print(gpu_monitor.info())
