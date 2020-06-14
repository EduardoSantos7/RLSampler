import random as rd
import pandas as pd


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

    def reset(self, random=True):
        if not random:
            if self.type == BOTH:
                return (1, 0)
            elif self.type == SAMPLES:
                return (1)
            else:
                return (0)

        # Get a random sample and frequency size from 1 to the number of rows
        random_sample_size = rd.randint(1, self.source.shape[0] + 1)
        random_frequency_size = rd.randint(1, self.source.shape[0] + 1)
        if self.type == BOTH:
            return (1, 0)
        elif self.type == SAMPLES:
            return random_sample_size
        else:  # Frequency
            return random_frequency_size


s = Sampler(pd.DataFrame())
print(s.reset())
