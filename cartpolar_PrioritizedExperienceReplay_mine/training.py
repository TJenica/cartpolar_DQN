import torch
import numpy as np
import sys
import pylab
import gym
from net import DQN

EPISODES = 200
MEMORYSIZE = 2000

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    env = env.unwrapped
    dqn = DQN(env.observation_space.shape[0], env.action_space.n, prioritized=True)
    score = 0
    scores, episodes = [], []
    for i in range(EPISODES):
        state = env.reset()

        done = False
        while not done:
            env.render()
            action = dqn.action_select(state)
            next_state, reward, done, _ = env.step(action)

            x, x_dot, theta, theta_dot = next_state  # ?
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            new_r = r1 + r2

            dqn.store(state, action, new_r, next_state)
            score += new_r
            state = next_state
            if dqn.memory_index > MEMORYSIZE:
                dqn.learning()
            if done:
                print('episode%s---reward_sum: %s' % (i, round(score, 2)))
                break


