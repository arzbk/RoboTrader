import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

# Responsible for mapping state observations to actions.
class Actor(nn.Module):

    # Initialize nn.Module and setup network
    def __init__(self, state_params, action_params):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_params, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_params)

        
    def forward(self, state):
        a = F.relu(self.layer_1(state))
        a = F.relu(self.layer_2(a))
        return torch.tanh(self.layer_3(a))


# Estimates future reward given current action, state, and policy
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# The main implementation of the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm
class TD3(object):
 
    
    def __init__(
            self,
            state_dim,
            action_dim,
            action_range,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
        ):

        # Define where the data is being processed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device.type} for compute.")

        # Calculate parameter load of actions and observations
        self.state_params = np.prod([s_dim for s_dim in state_dim])
        self.action_params = np.prod([a_dim for a_dim in action_dim])

        # Load networks into memory and leverage Adam for optimizations
        self.actor = Actor(self.state_params, self.action_params).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_loss = None

        self.critic = Critic(self.state_params, self.action_params).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic_loss = None

        self.action_range = action_range
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    # Generates action from actor policy given state
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        # Scales action to fill action space for environment
        for i in range(0, len(action)):
            low, high = self.action_range[i]
            action[i] = low + (action[i] + 1.0) * 0.5 * (high - low)

        return action


    # Pass however many np.arrays to this and it will prep them for the networks
    def prep_batch_data(self, *arrays):

        reshaped_arrays = []
        for arr in arrays[0]:


            # Get the size of the first dimension (batch size)
            batch_size = arr.shape[0]

            # Calculate the product of dimensions excluding the first dimension
            flattened_size = np.prod(arr.shape[1:])

            # Reshape the array, preserving the first dimension and flattening the rest
            reshaped_array = arr.reshape((batch_size, flattened_size))
            reshaped_arrays.append(torch.FloatTensor(reshaped_array).to(self.device))

        return reshaped_arrays

    def train(self, replay_buffer, batch_size=256):

        self.total_it += 1

        # Sample replay buffer - then reshape to: (batch_size, data)
        state, action, next_state, reward, not_done = self.prep_batch_data(replay_buffer.sample(batch_size))

        # Select action according to policy and add clipped noise
        with torch.no_grad():

            next_action = self.actor_target(next_state)

            # Add noise and compute next action clipping on action_range for each dimension
            for i in range(0, len(action[0])):

                low, high = self.action_range[i]

                noise = (
                    torch.randn_like(action[i]) * self.policy_noise[i]
                ).clamp(-self.noise_clip[i], self.noise_clip[i])

                next_action[i] = (
                    next_action[i] + noise
                ).clamp(low, high)

        # Compute the target Q value
        q_vals = self.critic_target(next_state, next_action)
        target_q = reward + not_done * self.discount * torch.min(*q_vals)

        # Get current Q estimates
        q_vals = self.critic(state, action)

        # Compute critic loss across all critics
        self.critic_loss = 0
        for q_val in q_vals:
            self.critic_loss = F.mse_loss(q_val, target_q) + self.critic_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        # Only update policy every n iterations - where n = policy_freq
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            self.actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

