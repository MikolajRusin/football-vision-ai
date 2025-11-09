from dataclasses import dataclass
from collections import deque
from pathlib import Path
import datetime as dt
import torch.nn as nn
import torch
import os

from torch.utils import checkpoint

@dataclass
class ModelCheckpointManager:
    checkpoint_dir_path: str | Path | None = None
    max_checkpoints: int = 5

    def __post_init__(self):
        if self.checkpoint_dir_path is not None:
            self.checkpoint_dir_path = Path(self.checkpoint_dir_path)
            self.checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            cur_time = dt.datetime.now().strftime('%d%m%Y_%H%M%S')
            self.checkpoint_dir_path = Path(__file__).parents[2] / 'checkpoints' / f'run_{cur_time}'
            self.checkpoint_dir_path.mkdir(parents=True)

        self.checkpoints_queue = self._load_start_queue()

    def _load_start_queue(self):
        checkpoints_queue = deque([])
        checkpoint_files = os.listdir(self.checkpoint_dir_path)
        if len(checkpoint_files) > 0:
            datestamps = self._get_file_modification_time(checkpoint_files)
            args_sorted_dt = [i[0] for i in sorted(enumerate(datestamps), key=lambda x: x[1])]
            checkpoints_queue.extend([checkpoint_files[i] for i in args_sorted_dt])
        return checkpoints_queue

    def _get_file_modification_time(self, files: list[str]):
        datestamps = []
        for file in files:
            file_path = self.checkpoint_dir_path / file
            timestamp = os.path.getmtime(str(file_path))
            datestamp = dt.datetime.fromtimestamp(timestamp)
            datestamps.append(datestamp)
        return datestamps[:self.max_checkpoints]

    def _set_checkpoint_name(self, epoch: int | None = None, iteration: int | None = None):
        cur_time = dt.datetime.now().strftime('%d%m%Y_%H%M%S')
        checkpoint_name = f'checkpoint'
        checkpoint_name += f'_e{epoch}' if epoch is not None else ''
        checkpoint_name += f'_i{iteration}' if iteration is not None else ''
        checkpoint_name += f'_{cur_time}.pth'
        return checkpoint_name

    def _update_queue(self, checkpoint_name: str):
        self.checkpoints_queue.append(checkpoint_name)
        if len(self.checkpoints_queue) > self.max_checkpoints:
            oldest_checkpoint_file = self.checkpoints_queue.popleft()
            (self.checkpoint_dir_path / oldest_checkpoint_file).unlink()

    def save_checkpoint(
        self,
        model: nn.Module,
        checkpoint_name: str | None = None, 
        epoch: int | None = None,
        iteration: int | None = None
    ):
        if checkpoint_name is None:
            checkpoint_name = self._set_checkpoint_name(epoch, iteration)
        checkpoint_path = self.checkpoint_dir_path / checkpoint_name

        checkpoint = {
            # 'model_state_dict': model.model.state_dict() if hasattr(model, 'model') else model.state_dict(),
            'epoch': epoch,
            'iteartion': iteration
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint has been saved as: {checkpoint_path}')

        self._update_queue(checkpoint_name)
        