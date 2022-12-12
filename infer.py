from argparse import ArgumentParser
import yaml
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, f1_score

import torch
import hydra

from dataset import KLAID_dataset
from model import DQNClassification
from policy_model import PolicyGradientClassification

def get_network(net_name, checkpoint_path, config):
    if 'DQN' in net_name:
        model = DQNClassification(hparams=config.model, run_mode='test')
        model.to('cpu')
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")), strict=False)
    else:
        model = PolicyGradientClassification(hparams=config.model, run_mode='test')
        model.to('cpu')
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")), strict=False)


    return model


@hydra.main(version_base=None, config_path='conf', config_name='infer_policy')
def main(args):
    # get RL network
    model = get_network(args.net_name, args.checkpoint_path, args)
    if args.net_name == 'DQN':
        net = model.classification_model
    elif args.net_name == 'DQN_double' or args.net_name == 'DQN_dueling':
        net = model.target_model
    else:
        net = model.p_net

    dataset = model.dataset
    label_list = dataset.class_list
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
    conf_matrix = confusion_matrix(y_true=trues, y_pred=preds, labels=label_list)
    print(conf_matrix)


if __name__=='__main__':
    main()