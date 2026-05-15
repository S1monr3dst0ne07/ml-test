from __future__ import annotations

import ctypes
import os


path = os.path.join(os.path.dirname(__file__), 'nn.so')
main = ctypes.CDLL(path)


class data_point_t(ctypes.Structure):
    _fields_ = (
        ('inputs',  ctypes.POINTER(ctypes.c_float)),
        ('outputs', ctypes.POINTER(ctypes.c_float)),
    )

class train_data_t(ctypes.Structure):
    _fields_ = (
        ('input_count',  ctypes.c_size_t),
        ('output_count', ctypes.c_size_t),
        ('count',        ctypes.c_size_t),
        ('data',         ctypes.POINTER(data_point_t)),
    )

class neuron_t(ctypes.Structure):
    _fields_ = (
        ('act',   ctypes.c_float),
        ('delta', ctypes.c_float),
        ('bias',  ctypes.c_float),
        ('weights_count', ctypes.c_size_t),
        ('weights',       ctypes.POINTER(ctypes.c_float)),
    )

    def json(self):
        return {
            'bias': self.bias,
            'weights': [
                self.weights[i] for i 
                in range(self.weights_count)
            ]
        }

class layer_t(ctypes.Structure):
    _fields_ = (
        ('count', ctypes.c_size_t),
        ('ns',    ctypes.POINTER(neuron_t)),
    )

    def json(self):
        return [
            self.ns[i].json() for i 
            in range(self.count)
        ]

class net_t(ctypes.Structure):
    _fields_ = (
        ('layers_conf', ctypes.POINTER(ctypes.c_size_t)),
        ('count', ctypes.c_size_t),
        ('ls', ctypes.POINTER(layer_t)),
        ('input_count',   ctypes.c_size_t),
        ('output_count',  ctypes.c_size_t),
        ('output_buffer', ctypes.POINTER(ctypes.c_float)),
    )

    def json(self):
        return [
            self.ls[i].json() for i 
            in range(self.count)
        ]




from dataclasses import dataclass as dc
from dataclasses import field


@dc
class DataPoint:
    inputs  : list[float]
    outputs : list[float]

    def load(self):
        return data_point_t(
            inputs  = (ctypes.c_float * len(self.inputs)) (*self.inputs),
            outputs = (ctypes.c_float * len(self.outputs))(*self.outputs),
        )


@dc
class TrainData:
    points : list[DataPoint] = field(default_factory=lambda: [])

    input_count  : int | None = None
    output_count : int | None = None

    def __iadd__(self, point):
        self.points.append(point)
        self._sync_dims()
        return self

    def _sync_dims(self):
        for point in self.points:
            if self.input_count  is None: self.input_count  = len(point.inputs)
            if self.output_count is None: self.output_count = len(point.outputs)

            assert self.input_count  == len(point.inputs)
            assert self.output_count == len(point.outputs)

    def load(self):
        self._sync_dims()

        count = len(self.points)
        return train_data_t(
            input_count  = self.input_count,
            output_count = self.output_count,
            count = count,
            data = (data_point_t * count)(*[
                p.load() for p in self.points
            ])
        )



main.nn_create_net.restype = net_t
main.nn_create_net.argtypes = (ctypes.POINTER(ctypes.c_size_t),)
def create_net(config : list[int]) -> net_t:
    config.append(0)

    return main.nn_create_net(
        (ctypes.c_size_t * len(config))(*config)
    )

main.nn_train.restype = ctypes.c_float
main.nn_train.argtypes = (net_t, train_data_t, ctypes.c_float)
def train(net : net_t, td : train_data_t, rate : float) -> float:
    return main.nn_train(net, td, rate)


main.nn_loss.restype = ctypes.c_float
main.nn_loss.argtypes = (net_t, train_data_t)
def loss(net : net_t, td : train_data_t) -> float:
    return main.nn_loss(net, td)


main.nn_forward.restype = ctypes.POINTER(ctypes.c_float)
main.nn_forward.argtypes = (net_t, ctypes.POINTER(ctypes.c_float))
def forward(net : net_t, inputs : list[float]) -> list[float]:
    output = main.nn_forward(
        net,
        (ctypes.c_float * len(inputs))(*inputs)
    )

    return output


main.nn_delta.restype = ctypes.c_float
main.nn_delta.argtypes = (net_t,)
def delta(net : net_t):
    return main.nn_delta(net)






