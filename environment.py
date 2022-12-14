import random
from typing import List, Dict
import gym
from gym.utils import seeding
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset


class ClassifyEnv(gym.Env):
    def __init__(self, run_mode: str, dataset: Dataset):
        self.run_mode = run_mode    # run_mode: either train or test
        self.dataset = dataset
        self.env_data = [(data['encoded_output'], data['encoded_attention_mask']) for data in self.dataset]
        # self.env_encoded_output = [data['encoded_output'] for data in self.dataset]
        # self.env_attention_mask = [data['encoded_attention_mask'] for data in self.dataset]
        # self.env_data = [[data['encoded_output'], data['encoded_attention_mask']] for data in self.dataset[:len(self.dataset)]][0]   # env_data : list of inputs
        self.answer = [data['law_service_id'] for data in self.dataset]             # answer : list of answers corresponding to the input

        # 논문과 달리 Multi-class classification 문제이기 때문에,
        # majority class인지 minority class인지 저장하는 list가 필요함
        majority_class, minority_class = self.dataset.get_major_minor_class()
        self.majorities = majority_class.keys()
        self.minorities = minority_class.keys()
        self.imb_rate = majority_class             # imb_rate : imbalanced rates for each majority class; dict {key: class_num / value : imb_rate}

        self.game_len = len(self.dataset)                              # length of this episode = the number of data
        self.num_classes = len(list(set(self.answer)))                 # num_classes : the number of classes
        self.action_space = gym.spaces.Discrete(self.num_classes)
        print(self.action_space)
        self.step_id = 0
        self.y_preds = []     # y_preds : list of predictions

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, prediction):
        # Input : model's prediction value
        # Output : data, reward, is_terminal, info
        self.y_preds.append(prediction)
        info = {}
        terminal = False
        answer_t = self.answer[self.step_id]  # answer of this step
        if prediction == answer_t:
            if answer_t in self.minorities:
                reward = 1
            else:
                reward = 1.0 * self.imb_rate[answer_t]
        else:
            if answer_t in self.minorities:
                reward = -1
                if self.run_mode == 'train':
                    terminal = True
            else:
                reward = -1.0 * self.imb_rate[answer_t]
        self.step_id += 1

        if self.step_id == self.game_len - 1:
            y_true_cur = self.answer[:self.step_id]
            info['fmeasure'] = self.get_metrics(self.y_preds, y_true_cur)

            terminal = True  # end of step

        return (self.env_data[self.step_id][0], self.env_data[self.step_id][1]), reward, terminal, info

    def get_metrics(self, y_pred, y_true):
        # weighted f1-score
        f1 = f1_score(y_true, y_pred, average='weighted')
        return f1

    def reset(self):
        if self.run_mode == 'train':
            random.shuffle(self.env_data)
        self.step_id = 0
        self.y_preds = []

        return self.env_data[self.step_id]     # tuple


if __name__ == "__main__":
    from dataset import KLAID_dataset
    dataset = KLAID_dataset(split='test')
    env = ClassifyEnv('train', dataset)
    print(len(env.answer))
    print(env.env_data[0])


