# code reference : keras-rl/rl/memory.py

from collections import deque

class RingBuffer(object):
    # Deque는 random access 가 느려서 RingBuffer를 쓴다고 하는데
    # data가 deque인데 뭐가 다른 거임?
    def __init__(self, max_len):
        self.max_len = max_len
        self.data = deque(maxlen=self.max_len)
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def append(self, v):
        self.data.append(v)

    def __getitem__(self,idx):
        if idx < 0 or idx >= self.length:
            raise KeyError("index out of range")
        return self.data[idx]


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.win_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observs = deque(maxlen=self.win_length)
        self.recent_terminals = deque(maxlen=self.win_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, ternimal, training=True):
        self.recent_observs.append(observation)
        self.recent_terminals.append(ternimal)

    def get_recent_state(self, current_observation):
        pass

class SequentialMemory(Memory):
    def __init__(self, limit, kwargs):
        super(SequentialMemory, self).__init__(**kwargs)

        self.limit = limit

        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

        self.num_entries = len(self.observations)

    def sample(self, batch_size, batch_idxs=None):
        assert self.num_entries >= self.win_length +2, "Not enough entries in the memory"

    def append(self, observation, action, reward, terminal, training=True):
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)

        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)