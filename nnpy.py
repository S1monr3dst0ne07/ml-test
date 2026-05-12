
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
        ('input_count',   ctypes.c_size_t),
        ('outputs_count', ctypes.c_size_t),
        ('count',         ctypes.c_size_t),
        ('data',          ctypes.POINTER(data_point_t)),
    )

class neuron_t(ctypes.Structure):
    _fields_ = (
        ('act',   ctypes.c_float),
        ('delta', ctypes.c_float),
        ('bias',  ctypes.c_float),
        ('weights_count', ctypes.c_size_t),
        ('weights',       ctypes.POINTER(ctypes.c_float)),
    )

class layer_t(ctypes.Structure):
    _fields_ = (
        ('count', ctypes.c_size_t),
        ('ns',    ctypes.POINTER(neuron_t)),
    )

class net_t(ctypes.Structure):
    _fields_ = (
        ('layers_conf', ctypes.POINTER(ctypes.c_size_t)),
        ('count', ctypes.c_size_t),
        ('ls', ctypes.POINTER(layer_t)),
        ('input_count',   ctypes.c_size_t),
        ('output_count',  ctypes.c_size_t),
        ('output_buffer', ctypes.POINTER(ctypes.c_float)),
    )



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




