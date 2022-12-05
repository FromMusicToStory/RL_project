from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import statistics
from collections import Counter


class KLAID_dataset(Dataset):
    def __init__(self, model_name='klue/roberta-base', split='all'):
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

    def split_dataset(self, dataset, test_size=0.1):
        dict_ = dataset.train_test_split(test_size=test_size, shuffle=True)
        return dict_['train'], dict_['test']

    def get_class_num(self):
        num_cls = Counter([data['laws_service_id'] for data in self.dataset])
        return num_cls

    def get_major_minor_class(self, divide_by='mean'):
        if divide_by == 'mean':
            majority_class = {}
            minority_class = {}
            class_num = self.get_class_num()
            for key, value in zip(class_num.keys(), class_num.values()):
                if value < statistics.mean(class_num.values()):
                    minority_class[key] = value
                else:
                    majority_class[key] = value

            majority_class = {key: value for key, value in sorted(majority_class.items(), key=lambda item: item[1], reverse=True)}
            minority_class = {key: value for key, value in sorted(minority_class.items(), key=lambda item: item[1], reverse=True)}
        return majority_class, minority_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        law_service_id = self.dataset[idx]['laws_service_id']
        law_service = self.dataset[idx]['laws_service']
        fact = self.dataset[idx]['fact']

        encoding = self.tokenizer.encode_plus(fact,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              truncation=True,
                                              padding='max_length',
                                              return_tensors='pt')

        return {'law_service_id': law_service_id,
                'law_service': law_service,
                'fact': fact,
                'encoded_output': encoding['input_ids'].flatten(),
                'encoded_attention_mask': encoding['attention_mask'].flatten()}


if __name__ == '__main__':
    dataset = KLAID_dataset('all')
    print(dataset[0])
    print(len(dataset))

    majority_class, minority_class = dataset.get_major_minor_class()
    print(majority_class)
    print(minority_class)