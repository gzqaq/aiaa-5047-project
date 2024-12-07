class Timer:
    avg_tm: float
    cnt: int

    def __init__(self) -> None:
        self.avg_tm = 0.0
        self.cnt = 0

    def update_average(self, elapsed_tm: float) -> float:
        self.cnt += 1
        self.avg_tm += (elapsed_tm - self.avg_tm) / self.cnt

        return self.avg_tm
