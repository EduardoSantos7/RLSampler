import time

from predictive.MLManager import MLManager
from preprocessor.preprocessor import Preprocessor
from data.DataManager import DataManager
from RL.environments.sampler import Sampler
from RL.agents.agent import QAgent

from matplotlib import pyplot as plt
import numpy as np
import pprint

# p = Preprocessor()
# t = time.perf_counter()
# p.process()
# print("total time: ", time.perf_counter() - t)


data = DataManager.get_joined_data()

env = Sampler(data)
agent = QAgent(env)
agent.process()
# pprint.pprint(agent.q_table)
# print(s.reset(random=False))
# s.step(2)
# s.step(2)
# s.step(0)
# s.step(5)
# s.step(5)
# s.step(3)
# s.step(2)
# s.step(2)
# s.step(0)
# s.step(5)
# s.step(5)
# s.step(3)
# s.step(2)
# s.step(2)
# s.step(0)
# s.step(5)
# s.step(5)
# s.step(3)
# s.step(3)
# print(s.get_current_sample_frequency())
# print(s.get_samples_taken())

# google_model, X_train, X_test, y_train, y_test = MLManager.logistic_regression(
#     data)
# rl_model, _, _, _, _ = MLManager.logistic_regression(
#     s.get_samples_taken())

# reward = MLManager.compate_outpt(
#     google_model.predict(X_test), rl_model.predict(X_test))

# print(reward)

# print(google_model.predict(np.array([0.031947982202754625]).reshape(1, -1)))
# print(rl_model.predict(np.array([0.031947982202754625]).reshape(1, -1)))
