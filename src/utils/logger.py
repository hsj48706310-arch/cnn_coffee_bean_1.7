"""TensorBoard logger wrapper."""
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir: str, run_name: str | None = None):
        run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / run_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.path.as_posix())

    def log(self, scalars: dict, step: int) -> None:
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        self.writer.close()
