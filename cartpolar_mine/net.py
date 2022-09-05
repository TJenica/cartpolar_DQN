import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MEMORYSIZE = 2000
MINIBATCH = 32


class QNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        # super(QNetwork, self).__init__()
        super(QNetwork, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.hidden_network1 = nn.Linear(n_state, 128)
        self.hidden_network1.weight.data.normal_(0, 0.1)
        self.hidden_network2 = nn.Linear(128, 256)
        self.hidden_network2.weight.data.normal_(0, 0.1)
        self.output_network = nn.Linear(256, n_action)
        self.output_network.weight.data.normal_(0, 0.1)

    def forward(self, state):
        output = F.relu(self.hidden_network1(state))
        output = F.relu(self.hidden_network2(output))
        output = self.output_network(output)
        return output


class DQN(nn.Module):
    def __init__(self, n_state, n_action, epsilon = 0.9, discount = 0.9, learning_rate = 0.01):
        # super(QNetwork, self).__init__()
        super(DQN, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.epsilon = epsilon
        self.discount = discount
        self.learning_rate = learning_rate
        self.value_network, self.target_network = QNetwork(n_state, n_action), QNetwork(n_state, n_action)
        self.memory_index = 0
        self.memory = np.zeros((MEMORYSIZE, 2 * n_state + 1 + 1))
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.value_network.parameters(), lr = learning_rate)

    def store(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_index % MEMORYSIZE
        self.memory[index, :] = transition
        self.memory_index += 1

    def action_select(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.uniform() < self.epsilon:
            action_value = self.value_network.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_action)
        return action

    def learning(self):
        sample_index = np.random.choice(MEMORYSIZE, MINIBATCH)
        sample_transition = self.memory[sample_index, :]
        sample_state = torch.FloatTensor(sample_transition[:, 0: self.n_state])
        sample_action = torch.LongTensor(sample_transition[:, self.n_state:self.n_state + 1].astype(int))
        sample_reward = torch.FloatTensor(sample_transition[:, self.n_state + 1:self.n_state + 2])
        sample_next_state = torch.FloatTensor(sample_transition[:, -self.n_state])

        Q_value = self.value_network(sample_state).gather(1, sample_action)
        Q_nextvalue = self.target_network(sample_next_state).detach()
        Q_target = sample_reward + self.discount * Q_nextvalue.max(1)[0].view(MINIBATCH, 1)
        loss = self.loss(Q_target, Q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()








