import numpy as np
from typing import List, Dict
import gym
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset

class ClassifyEnv(gym.Env):
    def __init__(self, run_mode: str, dataset : Dataset):
        # run_mode : either train or test
        # imb_rate : imbalanced rates for each majority class, dict, key: class_num / value : imb_rate
        # env_data : list of inputs
        # answer : list of answers corresponding to the input
        # id :
        # game_len : length of this episode / the number of data
        # num_classes : the number of classes
        # action_space
        # step_id
        # y_pred : List
        self.run_mode = run_mode  # either train or test

        self.dataset = dataset
        self.answer = [data['law_service_id'] for data in self.dataset]

        # 논문과 달리 Multi-class classification 문제이기 때문에,
        # majority class인지 minority class인지 저장하는 list가 필요
        majority_class, minority_class = self.dataset.get_major_minor_class()
        self.majorities = majority_class.keys()
        self.minorities = minority_class.keys()
        self.imb_rate =  majority_class # how imbalanced is this dataset

        self.id = np.arange(len(self.dataset))

        self.game_len = len(self.dataset)
        self.num_classes = len(self.answer)
        self.action_space = gym.spaces.Discrete(self.num_classes)
        print(self.action_space)
        self.step_ind = 0
        self.y_preds = []

    def action_space(self):
        """
        1 for the minority class, 0 for the majority class.
        """
        return self.action_space

    def step(self, prediction):
        # Input : model's prediction value
        # Output : data, reward, is_terminal, info
        # Function :

        self.y_preds.append(prediction)

        y_true_cur = []  # 정답값
        info = {}
        terminal = False
        truncated = False
        answer_t = self.answer[self.id[self.step_ind]]  # answer of this step
        if prediction == answer_t:
            if answer_t in self.majorities:
                reward = 1
            else:
                reward = 1.0 * self.imb_rate[answer_t]
        else:
            if answer_t in self.majorities:
                reward = -1
                if self.run_mode == 'train':
                    truncated = True
            else:
                reward = -1.0 * self.imb_rate[answer_t]
        self.step_ind += 1

        if self.step_ind == self.game_len - 1:
            y_true_cur = self.answer[self.id]
            info['fmeasure'] = self.get_metrics(self.y_preds, y_true_cur[:self.step_ind])

            terminal = True  # end of step

        return self.dataset[self.id[self.step_ind]], reward, terminal, truncated, info

    def get_metrics(self, y_pred, y_true):
        # weighted f1-score
        f1 = f1_score(y_true, y_pred, average='weighted')
        return f1

    def reset(self):
        if self.run_mode == 'train':
            np.random.shuffle(self.id)
        self.step_ind = 0
        self.y_pred = []

        return self.dataset[self.id[self.step_ind]]
