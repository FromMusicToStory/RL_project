import torch
from hydra.utils import instantiate

class DQNLoss(object):
    def __init__(self, criterion, gamma):
        self.criterion = criterion
        self.gamma = gamma

    def __call__(self, batch, main_net, target_net):
        states, actions, rewards, next_states, terminals = batch
        state_action_values = main_net(input_ids=states[0], attention_mask=states[1]).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = target_net(input_ids=states[0], attention_mask=states[1]).max(1)[0]
            next_state_values[terminals] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * self.gamma + rewards

        return self.criterion(state_action_values, expected_state_action_values)

class DoubleDQNLoss(DQNLoss):
    def __init__(self, criterion, gamma):
        super(DoubleDQNLoss, self).__init__(criterion=criterion, gamma=gamma)

    def __call__(self,batch, main_net, target_net):
        states, actions, rewards, next_states, terminals = batch
        state_action_values = main_net(input_ids=states[0], attention_mask=states[1]).gather(1,actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_actions = main_net(input_ids=states[0], attention_mask=states[1]).max(1)[1]
            next_state_values = target_net(input_ids=states[0], attention_mask=states[1]).gather(1,next_state_actions.unsqueeze(-1)).squeeze(-1)
            next_state_values[terminals] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * self.gamma + rewards

        return self.criterion(state_action_values, expected_state_action_values)