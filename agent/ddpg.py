import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from agent.models import Actor, Critic

'''
This class implements a DDPG agent compatible with IsaacLab/SKRL framework.
'''

class DDPGAgent:
    def __init__(self, 
                 observation_dim, 
                 action_dim, 
                 max_action, 
                 gamma=0.99, 
                 tau=0.005, 
                 buffer_size=1000000, 
                 actor_learning_rate=1e-4, 
                 critic_learning_rate=1e-3, 
                 device=None,
                 seed=None):
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Parameters
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(observation_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(observation_dim, action_dim, max_action).to(self.device)
        
        self.critic = Critic(observation_dim, action_dim).to(self.device)
        self.critic_target = Critic(observation_dim, action_dim).to(self.device)
        
        # Copy weights to target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, observation):
        """Get action from actor network (inference mode)"""
        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        action = self.actor.forward(observation).detach().cpu().numpy()[0]
        return action
    
    def update(self, observations, actions, rewards, next_observations, dones):
        """
        Update actor and critic networks.
        
        @param observations: State batch
        @param actions: Action batch
        @param rewards: Reward batch
        @param next_observations: Next state batch
        @param dones: Done flags batch
        @return: (critic_loss, actor_loss)
        """
        
        # Convert to tensors
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_observations = torch.FloatTensor(next_observations).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Critic loss
        Qvals = self.critic.forward(observations, actions)
        next_actions = self.actor_target.forward(next_observations)
        next_Q = self.critic_target.forward(next_observations, next_actions.detach())
        Qprime = rewards + (1 - dones) * self.gamma * next_Q
        Qprime = Qprime.detach()
        critic_loss = self.critic_criterion(Qvals, Qprime)
        
        # Actor loss
        policy_loss = -self.critic.forward(observations, self.actor.forward(observations)).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return critic_loss.item(), policy_loss.item()

    def save_model(self, filename="ddpg_model.pth"):
        """Save agent networks and optimizers"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="ddpg_model.pth"):
        """Load agent networks and optimizers"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Model loaded from {filename}")
