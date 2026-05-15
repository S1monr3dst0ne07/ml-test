#!/usr/bin/python3

from dataclasses import dataclass as dc
import random
from nn import nnpy
import time
import numpy as np

random.seed(time.time_ns())

@dc
class petri:
    @dc
    class place:
        tokens     : int = 0 #number of token at place
        _buffer    : int = 0 #buffer for stepping
        _detractor : int = 0 
            #upon activation, the activation weight need to be take away from the token count of a place.
            #naturally, if two transitions activate on a location at the same time, the bigger detractor
            #need to be known, to compute the final value.

    @dc
    class conn:
        target : "petri.place"
        activation : int = 0 #number of token needing to be present in target for transition to fire
        receive    : int = 0 #number of token which are received by the place when the transition fires

    @dc
    class transition:
        conns : set["petri.connection"]

        def __iter__(self):
            return iter(self.conns)

    places      : dict[tuple[int,int    ], place]
    transitions : dict[tuple[int,int,int], transition]
    limit : int

    @classmethod
    def make(cls, x, y, n, limit):
        places = {}
        transitions = {}

        for xi in range(x):
            for yi in range(y):
                places[(xi, yi)] = cls.place()

        for xi in range(x-1):
            for yi in range(y-1):
                for ni in range(n):
                    coords = [
                        (xi + 0, yi + 0),
                        (xi + 1, yi + 0),
                        (xi + 0, yi + 1),
                        (xi + 1, yi + 1),
                    ]
                    conns = [cls.conn(places[coord]) for coord in coords]
                    transitions[(xi, yi, ni)] = cls.transition(conns)

        return cls(places, transitions, limit)
                
    def step(self):
        # forward pass
        for trans in self.transitions.values():
            if not all(
                    c.target.tokens >= c.activation 
                    for c in trans
            ): continue

            #activation
            for c in trans: 
                c.target._detractor = max(c.target._detractor, c.activation)
                c.target._buffer += c.receive

        # backward pass
        for place in self.places.values():
            place.tokens += place._buffer
            place.tokens -= place._detractor

            place.tokens = min(place.tokens, self.limit)

            place._buffer = 0
            place._detractor = 0






pn_width  = 3
pn_height = 3
pn_breath = 2

sample_size = 10
token_limit = 10
config_size = pn_width * pn_height * pn_breath * 4 * 2

distri = [0, 0, 1]
rate = 1e-2


def embed(pn):
    vector = []
    for x in range(pn_width-1):
        for y in range(pn_height-1):
            for n in range(pn_breath):
                trans = pn.transitions[(x, y, n)]
                for d in range(4):
                    vector.append(trans.conns[d].activation / token_limit)
                    vector.append(trans.conns[d].receive    / token_limit)

    return vector

def disembed(vector):
    pn = petri.make(pn_width, pn_height, pn_breath, token_limit)
    for x in range(pn_width-1):
        for y in range(pn_height-1):
            for n in range(pn_breath):
                trans = pn.transitions[(x, y, n)]
                for d in range(4):
                    trans.conns[d].activation = int(vector.pop(0) * token_limit)
                    trans.conns[d].receive    = int(vector.pop(0) * token_limit)

    return pn



def generate(): 
    pn = petri.make(pn_width, pn_height, pn_breath, token_limit)
    pn.places[(0, 0)].tokens = 1

    # randomize weights
    for trans in pn.transitions.values():
        for conn in trans.conns:
            conn.activation = random.choice(distri)
            conn.receive    = random.choice(distri)

    # !!! HYPERPARAMETER !!!
    prope_coord = (pn_width-1, pn_height-1)

    # record sample
    sample = []
    for _ in range(sample_size):
        pn.step()
        v = pn.places[prope_coord].tokens
        sample.append(v)

    return sample, embed(pn)


def batch(batch_size):
    points = []
    while len(points) < batch_size:
        sample, vector = generate()
        if all(x == 0 for x in sample):
            continue

        points.append(nnpy.DataPoint(
            [x / token_limit for x in sample],
            vector
        ))

    return nnpy.TrainData(points = points)

nn = nnpy.create_net([sample_size, 20, 20, 20, config_size])

i = 0
batch_size = 200
train = batch(batch_size).load()
while True:
    losses = []
    for _ in range(100):
        losses.append(nnpy.train(nn, train, rate))
        i += 1

    loss = np.average(losses)
    if loss < 0.01:
        train = batch(batch_size).load()
        print("--- new batch ---")

    delta = nnpy.delta(nn)
    print(f'epoch: {i}, loss: {loss}, delta: {delta}')











