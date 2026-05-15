#!/usr/bin/python3

from nn import nnpy
import time
import json
import random

train = nnpy.TrainData(points = [
    nnpy.DataPoint([0.0, 0.0], [0.0]),
    nnpy.DataPoint([1.0, 0.0], [1.0]),
    nnpy.DataPoint([0.0, 1.0], [1.0]),
    nnpy.DataPoint([1.0, 1.0], [0.0]),
])

random.seed(time.time_ns())

rate = 10
net = nnpy.create_net([2, 2, 1])

loaded_train = train.load()
for i in range(10000):
    loss = nnpy.train(net, loaded_train, rate)

    if i % 1000 == 0:
        print(f"epoch: {i}, loss: {loss}, delta: {nnpy.delta(net)}")

cost = nnpy.loss(net, loaded_train)
print(f"cost: {cost}")


print("----------- full net -----------")
for point in train.points:
    output = nnpy.forward(net, point.inputs)[0]
    x, y = point.inputs
    print(f"{x} | {y} = {output}")


with open('xor.json', 'w') as f:
    json.dump(net.json(), f, indent=2)


