import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc1 = nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(envs.single_action_space.shape))
        self.register_buffer(
            "action_scale", torch.tensor((envs.action_space.high - envs.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((envs.action_space.high + envs.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc1 = nn.Linear(np.array(envs.single_observation_space.shape).prod() + np.prod(envs.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TD3:
    def __init__(self,
                 envs,
                 device,
                 tau,
                 gamma,
                 noise_clip,
                 policy_frequency,
                 policy_noise,
                 replay_buffer,
                 actor_learning_rate,
                 critic_learning_rate,
                 exploration_noise,
                 actions_low,
                 actions_high,
                 batch_size):

        # Learning params
        self.tau = tau
        self.gamma = gamma
        self.noise_clip = noise_clip
        self.policy_frequency = policy_frequency
        self.policy_noise = policy_noise
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.exploration_noise = exploration_noise
        self.actions_low = actions_low
        self.actions_high = actions_high
        self.batch_size = batch_size

        self.replay_buffer = replay_buffer

        self.actor = Actor(envs).to(device)
        self.qf1 = QNetwork(envs).to(device)
        self.qf2 = QNetwork(envs).to(device)
        self.qf1_target = QNetwork(envs).to(device)
        self.qf2_target = QNetwork(envs).to(device)
        self.target_actor = Actor(envs).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=critic_learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=actor_learning_rate)


    def get_actions(self, obs, exploration_noise):
        with torch.no_grad():
            actions = self.actor(torch.Tensor(obs).to(self.device))
            actions += torch.normal(0, self.actor.action_scale * exploration_noise)
            actions = actions.cpu().numpy().clip(self.actions_low, self.actions_high)

    def train_on_batch(self, data, global_step):
        data = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            clipped_noise = (torch.randn_like(data.actions, device=self.device) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            ) * self.target_actor.action_scale

            next_state_actions = (self.target_actor(data.next_observations) + clipped_noise).clamp(
                self.actions_low, self.actions_high
            )
            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if global_step % self.policy_frequency == 0:
            actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
