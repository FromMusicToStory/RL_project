from argparse import ArgumentParser
import yaml
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, f1_score

import torch
import hydra

from dataset import KLAID_dataset
from model import DQNClassification
from policy_model import PolicyGradientClassification
from network import Classifier, DuelingClassifier, PolicyNet

def get_network(net_name, checkpoint_path, config):
    # hparams = yaml.load(open('conf/dqn.yaml', 'r'), Loader=yaml.FullLoader)
    # hparams['gpu'] = ['cpu']
    # hparams['model_name'] = 'klue/roberta-base'
    # hparams['net'] = Classifier(model_name=hparams['model_name'], num_classes=10)

    if 'DQN' in net_name:
        model = DQNClassification(hparams=config.model, run_mode='test')
        model.to('cpu')
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")), strict=False)
    else:
        model = PolicyGradientClassification.load_from_checkpoint(checkpoint_path)
        model.to('cpu')
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")), strict=False)

    network = None
    if 'DQN' == net_name:
        network = model.classification_model
    elif 'DQN_double' == net_name or 'DQN_dueling':
        network = model.target_model
    else:
        network = model.p_net
    print(network)
    return network, model.dataset

def pred_to_label(pred):
    pass

@hydra.main(version_base=None, config_path='conf', config_name='infer_dqn')
def main(args):
    # get RL network
    net, dataset = get_network(args.net_name, args.checkpoint_path, args)

    # infer
    trues, preds = [], []
    print("\nstart infer")
    for i, data in tqdm(enumerate(dataset)):  # get dataset
        input_ids, attention_mask, true_label = data['encoded_output'], data['encoded_attention_mask'], data['law_service_id']
        prediction = net(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
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
    main()
