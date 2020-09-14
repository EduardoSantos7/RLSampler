import random as rd
import pandas as pd

from predictive.MLManager import MLManager


# Types of state in env.
BOTH = 'both'
SAMPLES = 'samples'
FREQUENCY = 'frequency'

# Enum Actions
DECREASE_SAMPLES = 0
KEEP_SAME_SAMPLES = 1
INCREASE_SAMPLES = 2
DECREASE_FREQUENCY = 3
KEEP_SAME_FREQUENCY = 4
INCREASE_FREQUENCY = 5


class Sampler:

    def __init__(self, source, type_=BOTH, samples=1, frequency=0):
        """Init the env.

        Args:
            source (Pandas DF): The original data
            type (string): Indicates what should be used as state.
                           Defaults to BOTH. Other options are: samples or frequency
            samples (int, optional): indicates the number of continuos samples. Defaults to 1.
            frequency (int, optional): Indicates the number of spaces between sample groups.
                                       Defaults to 0.
        """

        if not isinstance(source, pd.DataFrame):
            raise Exception("Source must be a Pandas DataFrame")

        self.samples = samples
        self.frequency = frequency
        self.source = source
        self.type = type_
        self.data = None
        self.observation_space = 2 if self.type == BOTH else 1
        self.action_space = 6 if self.type == BOTH else 3
        self.samples_taken = []
        self.current_source_ptr = 0
        self.MIN_SAMPLES = 1
        self.MAX_SAMPLES = len(self.source)
        self.MIN_FREQUENCY = 0
        self.MAX_FREQUENCY = len(self.source)
        self.base_model, X_train, X_test, y_train, y_test = MLManager.logistic_regression(
            self.source)

    def reset(self, random=True):
        self.current_source_ptr = 0

        if not random:
            if self.type == BOTH:
                return (self.MIN_SAMPLES, self.MIN_FREQUENCY)
            elif self.type == SAMPLES:
                self.samples = self.MIN_SAMPLES
                return (self.MIN_SAMPLES)
            else:
                self.frequency = self.MIN_FREQUENCY
                return (self.MIN_FREQUENCY)

        # Get a random sample and frequency size from 1 to the number of rows
        random_sample_size = rd.randint(1, self.source.shape[0] - 1)
        random_frequency_size = rd.randint(0, self.source.shape[0] - 1)
        if self.type == BOTH:
            self.samples = random_sample_size
            self.frequency = random_frequency_size
            return (random_sample_size, random_frequency_size)
        elif self.type == SAMPLES:
            self.samples = random_sample_siz
            return random_sample_size
        else:  # Frequency
            self.frequency = random_frequency_size
            return random_frequency_size

    def step(self, action):
        info = {}

        if action == DECREASE_SAMPLES:
            if self.samples > self.MIN_SAMPLES:
                self.samples -= 1
        elif action == KEEP_SAME_SAMPLES:
            pass
        elif action == INCREASE_SAMPLES:
            if self.samples < self.MAX_SAMPLES - 1:
                self.samples += 1
        elif action == DECREASE_FREQUENCY:
            if self.frequency > self.MIN_FREQUENCY:
                self.frequency -= 1
        elif action == KEEP_SAME_FREQUENCY:
            pass
        elif action == INCREASE_FREQUENCY:
            if self.frequency < self.MAX_FREQUENCY - 1:
                self.frequency += 1

        # With this approach each step the agent sampling
        self.take_sample()

        observation = self.get_observation()
        done = self.is_done()
        reward = self.compute_reward()
        info = {}

        return observation, reward, done, info

    def take_sample(self):
        sample = self.source.iloc[
            self.current_source_ptr: self.current_source_ptr + self.samples]
        next_pos = self.samples + self.frequency
        self.current_source_ptr += next_pos if next_pos < self.MAX_SAMPLES else self.MAX_SAMPLES
        self.samples_taken.append(sample)

    def get_samples_taken(self):
        return pd.concat(self.samples_taken)

    def get_current_sample_frequency(self):
        return self.samples, self.frequency

    def get_observation(self):
        return self.samples, self.frequency

    def is_done(self):
        return True if self.current_source_ptr >= self.MAX_SAMPLES - 1 else False

    def compute_reward(self):
        sampled_data = self.get_samples_taken()
        rl_model, _, _, _, _ = MLManager.logistic_regression(sampled_data)
        _, X_test, _, _ = MLManager.split_train_test(self.source)

        base_reward = MLManager.compate_output(
            self.base_model.predict(X_test), rl_model.predict(X_test))

        len_diff_discount = len(sampled_data) / len(self.source)

        return base_reward - len_diff_discount
