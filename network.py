import torch
import torch.nn as nn
from transformers import AutoModel


class Classifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.clf_layer = nn.Linear(self.model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.clf_layer(pooled_output)
        return logits


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from dataset import KLAID_dataset

    dataset = KLAID_dataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Classifier('klue/roberta-base', 117)
    output = model(next(iter(dataloader))['encoded_output'], next(iter(dataloader))['encoded_attention_mask'])
    print(output.shape)