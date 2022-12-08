from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import statistics
from collections import Counter
from tqdm import tqdm

class KLAID_dataset(Dataset):
    def __init__(self, model_name='klue/roberta-base', split='all', extract_class=True):
        self.dataset = load_dataset("lawcompany/KLAID", 'ljp')['train']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = 125

        train, test = self.split_dataset(self.dataset, test_size=0.1)

        if split == 'train':
            self.dataset = train
        elif split == 'test':
            self.dataset = test
        else:
            self.dataset = self.dataset

        if extract_class:
            per_class_num  = self.get_class_num()
            class_list = self.extract_class(per_class_num, num_to_left=10)
            self.dataset = self.dataset.filter(lambda x: x['laws_service_id'] in class_list)

        self.encodings = []
        print("dataset preparing for faster loading")
        for fact in tqdm(self.dataset['fact']):
            encoding = self.tokenizer.encode_plus(fact,
                                      add_special_tokens=True,
                                      max_length=self.max_len,
                                      truncation=True,
                                      padding='max_length',
                                      return_tensors='pt')
            self.encodings.append(encoding)

    def split_dataset(self, dataset, test_size=0.1):
        dict_ = dataset.train_test_split(test_size=test_size, shuffle=True)
        return dict_['train'], dict_['test']

    def get_class_num(self):
        per_class_num = Counter([data['laws_service_id'] for data in self.dataset])
        return per_class_num

    def extract_class(self, class_num, num_to_left):
        per_class_num = {key: value for key, value in sorted(class_num.items(), key=lambda item: item[1], reverse=True)}
        class_list = list(per_class_num.keys())[:num_to_left]
        return class_list

    def get_major_minor_class(self, divide_by='mean'):
        if divide_by == 'mean':
            majority_class = {}
            minority_class = {}
            per_class_num = self.get_class_num()
            for key, value in zip(per_class_num.keys(), per_class_num.values()):
                if value < statistics.mean(per_class_num.values()):
                    minority_class[key] = value / sum(per_class_num.values())
                else:
                    majority_class[key] = value / sum(per_class_num.values())

            majority_class = {key: value for key, value in sorted(majority_class.items(), key=lambda item: item[1], reverse=True)}
            minority_class = {key: value for key, value in sorted(minority_class.items(), key=lambda item: item[1], reverse=True)}
        return majority_class, minority_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        law_service_id = self.dataset[idx]['laws_service_id']
        law_service = self.dataset[idx]['laws_service']
        fact = self.dataset[idx]['fact']
        tmp = self.encodings[idx]
        if isinstance(tmp, list):
            tmp = tmp[0]
        encoded_output = tmp['input_ids']
        attention_mask = tmp['attention_mask']
        # encoded_output = [encoding['input_ids'].flatten() for encoding in self.encodings[idx]]
        # attention_mask = [encoding['attention_mask'].flatten() for encoding in self.encodings[idx]]

        return {'law_service_id': law_service_id,
                'law_service': law_service,
                'fact': fact,
                'encoded_output': encoded_output.flatten(),
                'encoded_attention_mask': attention_mask.flatten()}


if __name__ == '__main__':
    dataset = KLAID_dataset(split='test')
    print(dataset[0])
    print(len(dataset))

    majority_class, minority_class = dataset.get_major_minor_class()
    print(majority_class)
    print(minority_class)
