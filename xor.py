#!/usr/bin/python3

import nnpy
import time

data = nnpy.TrainData(points = [
    nnpy.DataPoint([0.0, 0.0], [0.0]),
    nnpy.DataPoint([1.0, 0.0], [1.0]),
    nnpy.DataPoint([0.0, 1.0], [1.0]),
    nnpy.DataPoint([1.0, 1.0], [0.0]),
]).load()

nnpy.seed(time.time_ns())

rate = 1
net = nnpy.create_net([2, 2, 1])

for i in range(10):
    loss = nnpy.train(net, data, rate)
    print(f"epoch: {i}, loss: {loss}")

cost = nnpy.loss(net, data)
print(f"cost: {cost}")

