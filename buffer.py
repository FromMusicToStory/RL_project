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

        def sample(self, batch_size: int) -> Tuple:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            states, actions, rewards, next_states, terminals = zip(*[self.buffer[idx] for idx in indices])
            return torch.tensor(states, dtype=torch.float), \
                   torch.tensor(actions), \
                   torch.tensor(rewards, dtype=torch.float), \
                   torch.tensor(next_states, dtype=torch.float), \
                   torch.tensor(terminals, dtype=torch.uint8)


class RLDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int = 32):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple]:
        while True:
            yield self.replay_buffer.sample(self.batch_size)
