from apollocaffe.layers.layer_headers import Layer, PyLayer, LossLayer
from IPython import embed

import numpy as np

class Index(PyLayer):
    def forward(self, bottom, top):
        data, indices = bottom
        index_data = indices.data.astype(int)
        top[0].reshape(indices.shape)
        top[0].data[...] = data.data[range(indices.shape[0]), index_data]

    def backward(self, top, bottom):
        data, indices = bottom
        index_data = indices.data.astype(int)
        data.diff[...] = 0
        data.diff[range(indices.shape[0]), index_data] = top[0].diff

class AsLoss(PyLayer):
    def __init__(self, name, **kwargs):
        PyLayer.__init__(self, name, dict(), **kwargs)

    def forward(self, bottom, top):
        top[0].reshape(bottom[0].shape)
        top[0].data[...] = bottom[0].data

    def backward(self, top, bottom):
        bottom[0].diff[...] = top[0].data

class Reduction(LossLayer):
    def __init__(self, name, axis, **kwargs):
        kwargs["axis"] = axis
        super(Reduction, self).__init__(self, name, kwargs)
