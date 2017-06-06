# -*- coding: utf-8 -*-
'''
mathematical utilities.

Copyright © 2016 Thomas Unterthiner.
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''

import numpy as np

def softmax(x, axis=1, out=None):
    ''' Calculates softmax over axis 1. '''
    shape = list(x.shape)
    shape[axis] = 1
    m = x.max(axis=axis).reshape(shape)
    out = x - m
    e = np.exp(out, out=out)
    e /= e.sum(axis=axis).reshape(shape)
    return e


def to_onehot(x, max_classes):
    n_samples = x.shape[0]
    out = np.zeros((n_samples, max_classes), dtype=x.dtype)
    out.fill(0.0)
    for i in range(n_samples):
        out[i, x[i]] = 1
    return out
