import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MEMORYSIZE = 2000
MINIBATCH = 32


class SumTree(object):
    # 建立tree和data。
    # 因为SumTree有特殊的数据结构
    # 所以两者都能用一个一维np.array来存储
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    # 当有新的sample时，添加进tree和data
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    # 当sample被train,有了新的TD-error,就在tree中更新
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # 根据选取的v点抽取样本
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


# 自己打一遍
# stored as ( s, a, r, s_ ) in SumTree
class Memory(object):
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    # 建立SumTree和各种参数
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    # 存储数据，更新SumTree
    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    # 抽取sample
    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    # train完被抽取的samples后更新在tree中的sample的priority
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


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
    def __init__(self, n_state, n_action, epsilon = 0.9, discount = 0.9, learning_rate = 0.01, prioritized = True):
        # super(QNetwork, self).__init__()
        super(DQN, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.epsilon = epsilon
        self.discount = discount
        self.learning_rate = learning_rate
        self.prioritized = prioritized
        self.train_start = 1000
        self.value_network, self.target_network = QNetwork(n_state, n_action), QNetwork(n_state, n_action)
        self.memory_index = 0
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.value_network.parameters(), lr = learning_rate)
        if self.prioritized:
            self.memory = Memory(capacity=MEMORYSIZE)
        else:
            self.memory = np.zeros((MEMORYSIZE, 2 * n_state + 1 + 1))
        self.update_target_model()

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_network.load_state_dict(self.value_network.state_dict())

    def store(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        if self.prioritized:
            self.memory.store(transition)
        else:
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
        if self.prioritized:
            tree_idx, sample_transition, ISWeights = self.memory.sample(MINIBATCH)
        else:
            sample_index = np.random.choice(MEMORYSIZE, MINIBATCH)
            sample_transition = self.memory[sample_index, :]
        sample_state = torch.FloatTensor(sample_transition[:, 0: self.n_state])
        sample_action = torch.LongTensor(sample_transition[:, self.n_state:self.n_state + 1].astype(int))
        sample_reward = torch.FloatTensor(sample_transition[:, self.n_state + 1:self.n_state + 2])
        sample_next_state = torch.FloatTensor(sample_transition[:, -self.n_state])

        Q_value = self.value_network(sample_state).gather(1, sample_action)
        Q_nextvalue = self.target_network(sample_next_state).detach()
        Q_target = sample_reward + self.discount * Q_nextvalue.max(1)[0].view(MINIBATCH, 1)

        if self.prioritized:
            abs_errors = torch.abs(Q_target - Q_value).data.numpy
            loss = (torch.FloatTensor(ISWeights) * self.loss(Q_target, Q_value)).mean()
            self.memory.batch_update(tree_idx, abs_errors)
        else:
            loss = self.loss(Q_target, Q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()








