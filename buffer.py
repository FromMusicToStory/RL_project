from collections import deque, namedtuple
from typing import Tuple, Iterator
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset


Transition = namedtuple(
    "Transition",
    field_names= ["state", "action", "reward", "next_state", "ternminal"]
)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, transition: Transition):
        # transition = (state, action, reward, next_state, terminal)
        self.buffer.append(transition)

    def sample(self) -> Tuple:
        idx = np.random.choice(len(self.buffer), replace=False)
        state, action, reward, next_state, terminal = self.buffer[idx]
        state = state[0]  # states : list of input_ids, attention_mask, only get input_ids
        next_state = next_state[0]
        return state, \
               torch.tensor(action), \
               torch.tensor(reward, dtype=torch.float), \
               next_state, \
               torch.tensor(terminal, dtype=torch.uint8)    # state, next state are already tensors



class RLDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer

    def __iter__(self) -> Iterator[Tuple]:
        while True:
            yield self.replay_buffer.sample()
