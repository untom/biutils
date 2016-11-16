# -*- coding: utf-8 -*-
'''
Plotting functions.

Copyright © 2016 Thomas Unterthiner.
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''

from __future__ import absolute_import, division, print_function

import numpy as np

# Importing matplotlib might fail under special conditions
# e.g. when using ssh w/o X11 forwarding
try:
    import matplotlib.pyplot as plt
except ImportError:
    import warnings
    warnings.warn("matplotlib unavailable")

def plot_color_images(data, nrows=8, ncols=8, axis=None,
                     local_norm="minmax", **kwargs):
    plot_images(data, nrows, ncols, is_color=True, axis=axis, local_norm=local_norm, **kwargs)

def plot_images(data, nrows, ncols, is_color=False, axis=None,
                local_norm="maxabs", **kwargs):
    ''' Plots several images stored in the rows of data.'''
    nchannels = 3 if is_color else 1
    ppi = int(np.sqrt(data.shape[-1]/nchannels) + 2)  # +2 for borders
    imgshape = (nrows*ppi, ncols*ppi, nchannels)
    # make sure border is black
    img = {"maxabs": lambda s: (data.min() / np.abs(data).max()) * np.ones(imgshape, dtype=data.dtype),
           "minmax": lambda s: np.zeros(imgshape, dtype=data.dtype),
           "none":   lambda s: np.ones(imgshape, dtype=data.dtype)*data.min()
            }[local_norm.lower()](None)
    if len(data.shape) < 3:
        data = data.reshape(data.shape[0], nchannels, ppi-2, ppi-2)
    n = min(nrows*ncols, data.shape[0])
    normfunc = {"maxabs": lambda d: d / np.abs(d).max(),
                "minmax": lambda d: (d - d.min()) / d.ptp(),
                "none":   lambda d: d}[local_norm.lower()]
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx >= n:
                break
            d = np.rollaxis(data[idx, ], 0, 3)
            d = normfunc(d)
            img[r*ppi+1:(r+1)*ppi-1, c*ppi+1:(c+1)*ppi-1] = d
            idx += 1
    if axis is None:
        fig = plt.figure(facecolor="black", **kwargs)
        fig.subplots_adjust(hspace=0, top=1, bottom=0,
                            wspace=0, left=0, right=1)
        axis = fig.gca()
    else:
        fig = None
    if is_color:
        axis.imshow(img, interpolation="none")
    else:
        axis.imshow(img.reshape(nrows*ppi, ncols*ppi),
                    interpolation="none", cmap="Greys_r")
    axis.axis("off")
    return fig
