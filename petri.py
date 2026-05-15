#!/usr/bin/python3

from dataclasses import dataclass as dc
import random


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

    places      : dict[tuple[int,int], place]
    transitions : dict[tuple[int,int], transition]


    @classmethod
    def make(cls, x=4, y=4, n=2):
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

        return cls(places, transitions)
                
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

            place._buffer = 0
            place._detractor = 0




def sample(selection, size=10): 
    n = petri.make()
    n.places[(0, 0)].tokens = 1

    # randomize weights
    for trans in n.transitions.values():
        for conn in trans.conns:
            conn.activation = random.choice(selection)
            conn.receive    = random.choice(selection)

    # record sample
    sample = []
    for _ in range(size):
        n.step()
        v = n.places[(3, 3)].tokens
        sample.append(v)

    return sample


s = sample([0, 0, 0, 1])
print(s)









