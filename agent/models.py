import torch

from skrl.models.torch import DeterministicMixin, Model

class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], 256),
            torch.nn.LayerNorm(256), # THE ARMOR
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_space.shape[0]),
            torch.nn.Tanh(),
        )

    def compute(self, inputs, role):
        states = torch.nan_to_num(inputs["states"], nan=0.0, posinf=10.0, neginf=-10.0)
        output = self.net(states)
        if torch.isnan(output).any():
            print(f"🚨 [ACTOR NaN] States: {states[0].detach().cpu().numpy()}")
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        return output, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0] + action_space.shape[0], 256),
            torch.nn.LayerNorm(256), # THE ARMOR
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        states = torch.nan_to_num(inputs["states"], nan=0.0, posinf=10.0, neginf=-10.0)
        actions = torch.nan_to_num(inputs["taken_actions"], nan=0.0, posinf=1.0, neginf=-1.0)
        sa = torch.cat([states, actions], dim=1)
        q_value = self.net(sa)
        if torch.isnan(q_value).any():
            print(f"🚨 [CRITIC NaN] Q-value exploded!")
        q_value = torch.nan_to_num(q_value, nan=0.0, posinf=1e3, neginf=-1e3)
        return q_value, {}