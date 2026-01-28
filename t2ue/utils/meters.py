from dataclasses import dataclass

@dataclass
class AvgMeter:
    total: float = 0.0
    count: int = 0

    def update(self, v: float, n: int = 1) -> None:
        self.total += float(v) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)