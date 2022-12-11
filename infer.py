from argparse import ArgumentParser

from sklearn.metrics import confusion_matrix, f1_score

from dataset import KLAID_dataset
from model import DQNClassification
from policy_model import PolicyGradientClassification

def get_network(net_name, checkpoint_path):
    net = None
    if 'DQN' in net_name:
        net = DQNClassification.load_from_checkpoint(checkpoint_path)
    else:
        net = PolicyGradientClassification.load_from_checkpoint(checkpoint_path)
    return net

def pred_to_label(pred):
    pass

def main(args):
    # get dataset
    dataset = KLAID_dataset(model_name="klue/roberta-base", split='test')

    # get RL network
    # net = get_network(args.net_name, args.checkpoint_path)

    # infer
    trues, preds = [], []
    net = None
    for i, data in enumerate(dataset):
        input_ids, attention_mask, true_label = data['encoded_output'], data['encoded_attention_mask'], data['law_service_id']
        prediction = net.infer(input_ids=input_ids, attention_mask=attention_mask)
        pred_label = pred_to_label(prediction)
        trues.append(true_label)
        preds.append(pred_label)

    # calculate metrics
    weighted = f1_score(trues, preds, average='weighted')
    f1 = f1_score(trues, preds, average='macro')
    conf_matrix = confusion_matrix(y_true=trues, y_pred=preds, labels=dataset.class_list)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--net_name", type=str, default="DQN", help="which RL algorithm, DQN, DoubleDQN, DuelingDQN, PolicyGradient")
    parser.add_argument("--checkpoint_path", type=str, required=True)

    args = parser.parse_args()
    main(args)