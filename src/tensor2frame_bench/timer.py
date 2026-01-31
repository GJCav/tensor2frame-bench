from time import perf_counter
from typing import Callable
from attrs import define, field

@define(kw_only=True)
class Timer:
    repeat: int = field(default=4)
    func: Callable = field()
    
    records: list[float] = field(init=False, factory=list)

    def run(self):
        self.records.clear()
        for _ in range(self.repeat):
            start = perf_counter()
            self.func()
            end = perf_counter()
            self.records.append(end - start)
    
    @property
    def avg(self) -> float:
        if not self.records:
            raise ValueError("No records found. Please run the timer first.")
        return sum(self.records) / len(self.records)
    
    def __str__(self) -> str:
        return (
            f"Timer(func={self.func.__name__}, "
            f"repeat={self.repeat}, "
            "records=[" + ", ".join(f"{r:.6f}s" for r in self.records) + "], "
            f"avg={self.avg:.6f}s)"
        )

