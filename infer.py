from argparse import ArgumentParser
import yaml
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, f1_score

import torch
from dataset import KLAID_dataset
from model import DQNClassification
from policy_model import PolicyGradientClassification
from network import Classifier, DuelingClassifier, PolicyNet

def get_network(net_name, checkpoint_path):
    hparams = yaml.load(open('conf/dqn.yaml', 'r'), Loader=yaml.FullLoader)
    hparams['gpu'] = ['cpu']
    hparams['model_name'] = 'klue/roberta-base'
    hparams['net'] = Classifier(model_name=hparams['model_name'], num_classes=10)

    if 'DQN' in net_name:
        net = DQNClassification(hparams=hparams, run_mode='test')
        net.to('cpu')
        net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")), strict=False)
    else:
        net = PolicyGradientClassification.load_from_checkpoint(checkpoint_path)
        net.to('cpu')
        net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")), strict=False)

    return net

def pred_to_label(pred):
    pass

def main(args):
    # get RL network
    net = get_network(args.net_name, args.checkpoint_path)

    # infer
    trues, preds = [], []
    print("\nstart infer")
    for i, data in tqdm(enumerate(net.dataset)):  # get dataset
        input_ids, attention_mask, true_label = data['encoded_output'], data['encoded_attention_mask'], data['law_service_id']
        prediction = net.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        pred_label = prediction.argmax(1).item()
        trues.append(true_label)
        preds.append(pred_label)

    # calculate metrics
    weighted = f1_score(trues, preds, average='weighted')
    print(weighted)
    f1 = f1_score(trues, preds, average='macro')
    print(f1)
    conf_matrix = confusion_matrix(y_true=trues, y_pred=preds, labels=net.dataset.class_list)
    print(conf_matrix)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--net_name", type=str, default="DQN", help="which RL algorithm, DQN, DoubleDQN, DuelingDQN, PolicyGradient")
    parser.add_argument("--checkpoint_path", type=str, required=True)

    args = parser.parse_args()

    main(args)