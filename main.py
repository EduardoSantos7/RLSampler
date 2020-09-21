import time

from preprocessor.preprocessor import Preprocessor
from RL.environments.sampler import Sampler
from predictive.MLManager import MLManager
from data.DataManager import DataManager
from utils.PlotUtils import PlotUtils
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
agent.process(episodes=1000)
PlotUtils.different_axis(agent.statics.get('episodes'), agent.statics.get(
    'rewards'), agent.statics.get('sample_size'), agent.statics.get('epsilon'))
# PlotUtils.plot_lines([agent.statics.get('episodes')], [
#  agent.statics.get('rewards')], ['Rewards per episode'])
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
