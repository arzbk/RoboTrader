import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

class Actor(nn.Module):
    def __init__(self, envs, debug=False):
        super().__init__()

        # Get input and output neuron counts
        input_neurons = np.array(envs.single_observation_space.shape).prod()
        output_neurons = np.array(envs.single_action_space.shape).prod()

        if debug:
            print(f"ACTOR NETWORK SHAPE: {input_neurons} -> ... -> {output_neurons}.")

        self.fc1 = nn.Linear(input_neurons, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, output_neurons)
        self.register_buffer(
            "action_scale", torch.tensor((envs.action_space.high - envs.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((envs.action_space.high + envs.action_space.low) / 2.0, dtype=torch.float32)
        )

        # Reduce overhead of checking "if debug" on every network call
        self.forward = self.debug_forward if debug else self.forward

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

    def debug_forward(self, x_in):

        # Check if input values are in proper range
        test_1 = x_in <= 1
        test_2 = x_in >= -1
        if not test_1.all().item() or not test_2.all().item():
            raise Exception(f"""Actor Network has received value that falls outside of bounds -1 and +1:
            - Raw Tensor: {x_in}""")

        x = F.relu(self.fc1(x_in))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))

        retval = x * self.action_scale + self.action_bias

        # Check if output values are in proper range
        test_1 = x <= 1
        test_2 = x >= -1
        if not test_1.all().item() or not test_2.all().item():
            raise Exception(f"""Actor Network has produced value that falls outside of bounds -1 and +1:
            - Raw Tensor: {x}
            - Scaled Tensor: {retval}""")

        return retval

class QNetwork(nn.Module):
    def __init__(self, envs, debug=False):
        super().__init__()

        # Get input and output neuron counts
        state_inputs = np.array(envs.single_observation_space.shape).prod()
        action_inputs = np.array(envs.single_action_space.shape).prod()
        output_neurons = 1

        if debug:
            print(f"CRITIC NETWORK SHAPE: {(state_inputs, action_inputs)} -> ... -> {output_neurons}.")

        self.debug = debug
        self.fc1 = nn.Linear(state_inputs + action_inputs, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_neurons)

        # Reduce overhead of checking "if debug" on every network call
        self.forward = self.debug_forward if debug else self.forward

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def debug_forward(self, x_in, a):

        # Check if input values are in proper range
        test_1 = x_in <= 1
        test_2 = x_in >= -1
        if not test_1.all().item() or not test_2.all().item():
            raise Exception(f"""Critic/Q Network has received value that falls outside of bounds -1 and +1:
            - Raw Tensor: {x_in}""")

        # Pass data through network layers
        x = torch.cat([x_in, a], 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Check if output values are in proper range
        test_1 = x <= 1
        test_2 = x >= -1
        if not test_1.all().item() or not test_2.all().item():
            raise Exception(f"""Critic/Q Network has produced value that falls outside of bounds -1 and +1: 
            - Raw Tensor: {x}""")

        return x


class BatchNormWrapper(nn.Module):
    def __init__(self, model):
        super(BatchNormWrapper, self).__init__()
        self.model = model

    def forward(self, x, a):
        return self.model(x, a)

    def switch_to_train_mode(self):
        self.train()

    def switch_to_eval_mode(self):
        self.eval()


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
                 batch_size,
                 max_grad_norm=2.0,
                 debug=False):

        # Learning params
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.noise_clip = noise_clip
        self.policy_frequency = policy_frequency
        self.policy_noise = policy_noise
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.exploration_noise = exploration_noise
        self.actions_low = envs.action_space.low
        self.actions_low_tensor = torch.from_numpy(envs.action_space.low).to(device)
        self.actions_high = envs.action_space.high
        self.actions_high_tensor = torch.from_numpy(envs.action_space.high).to(device)
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer

        # Hyperparams intended for mitigating exploding gradient issue
        self.batchnorm_learning_enabled = False
        self.max_grad_norm = max_grad_norm

        # Only need to enable debugging - if asked - on one of each network
        self.actor = Actor(envs, debug=debug).to(device)
        self.qf1 = BatchNormWrapper(QNetwork(envs, debug=debug).to(device))

        self.qf2 = BatchNormWrapper(QNetwork(envs).to(device))
        self.qf1_target = BatchNormWrapper(QNetwork(envs).to(device))
        self.qf2_target = BatchNormWrapper(QNetwork(envs).to(device))
        self.target_actor = Actor(envs).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict(), strict=False)
        self.qf1_target.load_state_dict(self.qf1.state_dict(), strict=False)
        self.qf2_target.load_state_dict(self.qf2.state_dict(), strict=False)
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=critic_learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=actor_learning_rate)

        # Used for tracking metrics during training
        self.qf1_a_values = None
        self.qf2_a_values = None
        self.qf1_loss = None
        self.qf2_loss = None
        self.qf_loss = None
        self.actor_loss = None


    def switch_to_train_mode(self):
        self.batchnorm_learning_enabled = True
        self.qf1.switch_to_train_mode()
        self.qf2.switch_to_train_mode()


    def switch_to_eval_mode(self):
        self.batchnorm_learning_enabled = False
        self.qf1.switch_to_eval_mode()
        self.qf2.switch_to_eval_mode()

    def evaluate(self, eval_env, seed):
        cum_reward = 0
        done = False
        obs, info = eval_env.reset(seed=seed, has_ui=False, new_dates=False, new_tickers=False)
        while not done:
            actions = self.get_actions(obs, add_noise=False)
            obs, reward, done, _, info = eval_env.step(actions)
            cum_reward += reward

            # Log key variables for debugging
            logging.debug(
                f"EVAL: STEP ++++++ \n"
                + f"- Actions = {actions}\n"
                + f"- obs = {obs}\n"
                + f"- reward = {reward}; cum_reward = {cum_reward}\n"
                + f"- Net worth = ${info['net_worth']:.2f}\n"
            )

        # Return key values
        profit = info['net_worth'] - eval_env.total_cash
        return cum_reward, profit

    def get_actions(self, obs, add_noise=True):
        with torch.no_grad():
            actions = self.actor(torch.Tensor(obs).to(self.device))
            if add_noise:
                actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
                actions = actions.cpu().numpy().clip(self.actions_low, self.actions_high)
                return actions
            else:
                actions = actions.cpu().numpy()
                return actions


    def train_on_batch(self, update_policy=False):
        data = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():

            clipped_noise = (torch.randn_like(data.actions, device=self.device) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            ) * self.target_actor.action_scale

            next_state_actions = (self.target_actor(data.next_observations) + clipped_noise).clamp(
                self.actions_low_tensor, self.actions_high_tensor
            )

            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        self.qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        self.qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        self.qf1_loss = F.mse_loss(self.qf1_a_values, next_q_value)
        self.qf2_loss = F.mse_loss(self.qf2_a_values, next_q_value)
        self.qf_loss = self.qf1_loss + self.qf2_loss

        # prepare for optimization
        self.q_optimizer.zero_grad()
        self.qf_loss.backward()

        # Clip gradients before performing the optimization step
        #torch.nn.utils.clip_grad_norm_(max_norm=self.max_grad_norm)

        # optimize
        self.q_optimizer.step()

        if update_policy:
            self.actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
