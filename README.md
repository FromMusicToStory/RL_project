# RL Project
Reinforcement Learning Term Project for Fall Semester 2022

This is our improved version of the paper [Deep Reinforcement Learning for Imbalanced Classification](https://arxiv.org/abs/1901.01379).

The original Github is [here](https://github.com/linenus/DRL-For-imbalanced-Classification).

We aimed to extend the original method to a Korean multi-class text classification task using a pre-trained BERT model.

We used the [KLAID](https://huggingface.co/datasets/lawcompany/KLAID) dataset.

## TO-DO LIST

- [ ]  논문 github 보고 Environment, Agent 등 전체 프레임워크 Pytorch 로 바꾸기
- [ ]  [Keras-RL](https://github.com/keras-rl/keras-rl) 패키지에서 사용할 것 Pytorch로 구현
- [x]  Classification Model을 Pretrained Model 을 Finetuning 하는 것으로 바꾸기
- [ ]  DQN 을 다른 걸로 바꾸기
- [x]  KLAID 데이터셋 프로세싱하는 dataset.py 만들기 (data loading 용) => hugginface dataset 사용하면 됨


## Installation

```
pip install -r requirements.txt
```

## Training

* run vanilla DQN
```
mkdir checkpoint
python train.py
```

* run Double DQN
```
python train.py --config-name double_dqn
```

* run Dueling DQN
```
python train.py --config-name dueling_dqn
```

## Commands for Experiment
* gamma
```
python train.py --config-name dqn model.loss.gamma=0.1 name=DQN_base_0.1
python train.py --config-name dqn model.loss.gamma=0.5 name=DQN_base_0.5
python train.py --config-name dqn model.loss.gamma=0.9 name=DQN_base_0.9
```
* Smooth L1 Loss
```
python train.py --config-name dqn model.loss.criterion._target_=torch.nn.SmoothL1Loss name=DQN_smooth
python train.py --config-name double_dqn model.loss.criterion._target_=torch.nn.SmoothL1Loss name=double_smooth
python train.py --config-name dueling_dqn model.loss.criterion._target_=torch.nn.SmoothL1Loss name=dueling_smooth
```
* Random Seed
```
python train.py --config-name dqn seed=1111 name=dqn_1111
python train.py --config-name dqn seed=2222 name=dqn_2222
python train.py --config-name dqn seed=3333 name=dqn_3333
python train.py --config-name dqn seed=4444 name=dqn_4444
python train.py --config-name dqn seed=5555 name=dqn_5555
python train.py --config-name dqn seed=666 name=dqn_666
python train.py --config-name dqn seed=777 name=dqn_777 
python train.py --config-name dqn seed=6203 name=dqn_6203 
python train.py --config-name dqn seed=3040 name=dqn_3040 
python train.py --config-name dqn seed=6427 name=dqn_6427
```
## References
[Pytorch-Lightning Tutorial For DQN](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/reinforce-learning-DQN.html)
