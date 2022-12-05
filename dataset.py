from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class KLAID_dataset(Dataset):
    def __init__(self):
        self.dataset = load_dataset("lawcompany/KLAID", 'ljp')['train']
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
        self.max_len = 125

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
    from collections import Counter

    dataset = KLAID_dataset()
    print(dataset[0])
    print(Counter([data['law_service_id'] for data in dataset]))