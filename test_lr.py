__author__ = 'baixiao'

import math

for i in xrange(100):
    max_learning_rate = 0.002
    min_learning_rate = 0.0001
    decay_speed = 200.0  # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    print (min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed))
