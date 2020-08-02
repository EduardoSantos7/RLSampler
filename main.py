import time

from predictive.MLManager import MLManager
from preprocessor.preprocessor import Preprocessor

p = Preprocessor()
t = time.perf_counter()
p.process()
print("total time: ", time.perf_counter() - t)
