import os
from itertools import product
import time
import pprint as pp
import numpy as np
import random
import pickle
from statistics import mean


class QAgent:

    def __init__(self, env, q_table="", render=True, debug=False):
        self.env = env
        self.debug = debug
        self.q_table = q_table
        self.rewards_per_episode = []

    def process(self, episodes=2000, gamma=0.99, alpha=0.01, epsilon=1.0, epsilon_decrease=.001, policy="e-greedy"):
        self.init_q_tabe()

        for episode in range(episodes):

            # Update epsilon each 5 episodes
            if episode and episode % 5 == 0:
                epsilon = max(0.05, epsilon - epsilon_decrease)
                print(f'''
                    Episode: {episode}
                    Rewards: {mean(self.rewards_per_episode or [0])}
                    Epsilon: {epsilon}
                    Samples Size: {len(self.env.get_samples_taken())}
                    ''')

            state = self.env.reset(random=False)
            done = False
            rewards = 0
            while not done:
                action = self.pick_action(
                    self.q_table[state], epsilon=epsilon, policy=policy)
                new_state, reward, done, info = self.env.step(action)
                rewards += reward

                next_action = self.pick_action(
                    self.q_table[new_state], policy="greedy")
                q_value = self.q_table[state][action]
                next_q_value = self.q_table[new_state][next_action]

                self.q_table[state][action] += alpha * \
                    (reward + (gamma*next_q_value) - q_value)

                # print(
                #     f'episode {episode} state {state} action {action} reward {reward} new_state {new_state} {next_action}')
                state = new_state

            self.rewards_per_episode.append(rewards)

        # self.save_q_table(policy, episode)

    def init_q_tabe(self, path=None):
        # If there's a path
        if path:
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
        else:
            # Create the table
            self.q_table = {}
            for prod in product(range(0, self.env.MAX_SAMPLES),
                                repeat=self.env.observation_space):
                self.q_table[prod] = np.zeros((self.env.action_space,))

    def pick_action(self, actions, epsilon=1, policy="e-greedy"):
        if policy == "e-greedy":
            if random.random() < epsilon:
                return random.randint(0, len(actions) - 1)
            else:
                return int(np.argmax(actions))

        if policy == "greedy":
            return int(np.argmax(actions))

        if policy == "soft_max":
            prob_t = [np.exp(q_value/epsilon) for q_value in actions]
            prob_t = np.true_divide(prob_t, sum(prob_t))
            rand_probability = random.random() / 3
            # Get a random action from a list of actions which
            # contain values greater than a random probability.
            # If the list is empty then pick a random action.
            gradient = [i for i, num in enumerate(
                prob_t) if num > rand_probability]

            if gradient:
                return random.choice(gradient)

            return int(np.argmax(actions))

    def save_q_table(self, policy, episodes):
        dimensions = f'{self.env.maze_view.goal[0] + 1}x{self.env.maze_view.goal[0] + 1}'
        rewards = mean(self.rewards_per_episode)

        with open(f"q_table/q_learning/{dimensions}/{policy}_E_{episodes}.pickle", "wb") as f:
            pickle.dump(self.q_table, f)
