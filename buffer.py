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

    def sample(self, batch_size) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, terminals = zip(*(self.buffer[idx] for idx in indices))

        return states, \
               torch.tensor(actions), \
               torch.tensor(rewards), \
               next_states, \
               torch.tensor(terminals)


class RLDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer, buffer_size: int):
        self.replay_buffer = replay_buffer
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, next_states, terminals = self.replay_buffer.sample(batch_size=self.buffer_size)
        for i in range(len(terminals)):
            yield states[i], actions[i], rewards[i], next_states[i], terminals[i]


