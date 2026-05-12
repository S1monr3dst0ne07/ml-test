#!/usr/bin/python3


from nn import nnpy
import imageio
import numpy as np

img = imageio.imread('data/8.png')


#import cv2

dims = (256 << 3, 256 << 3)
#fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#vid = cv2.VideoWriter('output.mp4', fourcc, 20, dims)

train = nnpy.TrainData(points = [
    nnpy.DataPoint(
        [x / 28, y / 28], [p / 256],
    )
    for x, r in enumerate(img)
    for y, p in enumerate(r)
])

rate = 1e-2

net = nnpy.create_net([2, 5, 5, 1])

loaded_train = train.load()


def render(net):
    out = np.ndarray((dims[0], dims[1], 3), dtype=np.uint8)
    for y in range(dims[1]):
        print(y / dims[1])
        for x in range(dims[0]):
            u = x / dims[0]
            v = y / dims[1]

            p = int(nnpy.forward(net, [u, v])[0] * 255)
            out[x][y] = [p, p, p]

    return out

i = 0
while True:
    try:
        loss = nnpy.train(net, loaded_train, rate)
        if i % 100 == 0:
            print(f"epoch: {i}, loss: {loss}")
            #out = render(net, i // 100)
            #vid.write(out)
        i += 1
    except KeyboardInterrupt:
        break

cost = nnpy.loss(net, loaded_train)
print(f"cost: {cost}")

#vid.release()

imageio.imwrite('out.png', render(net))

