#!/usr/bin/env python2

from lasagne import layers
from lasagne.nonlinearities import rectify
import scipy
import scipy.misc
import numpy as np
import theano
import theano.tensor as T
import matplotlib.image

img = np.load("bad_input.npy")
b, g, r = img[0,...], img[1,...], img[2,...]
rgb = np.asarray([r, g, b])
transp = np.transpose(rgb, (1, 2, 0))
scipy.misc.imsave("bad_input.png", transp)
