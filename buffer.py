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
        # state, next state are already tensors
        return torch.cat([x[0] for x in states]), \
               torch.tensor(actions), \
               torch.tensor(rewards), \
               torch.cat([x[0] for x in next_states]), \
               torch.tensor(terminals), torch.cat([x[1] for x in states]), torch.cat([x[1] for x in next_states])




class RLDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, next_states, terminals, cur_atts, next_atts = self.replay_buffer.sample(batch_size=self.batch_size)
        for i in range(len(terminals)):
            yield states[i], actions[i], rewards[i], next_states[i], terminals[i], cur_atts[i], next_atts[i]
