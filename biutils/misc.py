# -*- coding: utf-8 -*-
'''
Various utility functions

Copyright © 2016 Thomas Unterthiner.
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''

from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np


def generate_slices(n, slice_size, ignore_last_minibatch_if_smaller=False):
    """Generates slices of given slice_size up to n"""
    start, end = 0, 0
    for pack_num in range(int(n / slice_size)):
        end = start + slice_size
        yield slice(start, end, None)
        start = end
    # last slice might not be a full batch
    if not ignore_last_minibatch_if_smaller:
        if end < n:
            yield slice(end, n, None)


def download_file(urlbase, destination_dir, fname=None):
    ''' Downloads a file to a given destination directory.'''
    import sys
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    if sys.version_info < (3,):
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    if fname is not None:
        url = urlbase + fname
    else:
        url = urlbase
        fname = os.path.basename(url)

    dst = os.path.join(destination_dir, fname)
    urlretrieve(url, dst)
    return dst


def print_system_information(additional_modules=[]):
    '''Prints general system information.

    Prints host information as well as version information about some of the
    more important packages. This is useful in IPython notebooks.'''
    import sys, datetime, platform  # imported here for quicker imports
    host_info = (platform.node(), platform.platform())
    print("Host:               ", "%s: %s" % host_info)
    print("Date:               ", str(datetime.datetime.now()))
    print("Python version:     ", sys.version.replace("\n", "\n" + " "*21))

    repo_version = str(os.popen("git log | head -1").readline().strip())
    if not repo_version.startswith("fatal:"):
        print("repository version: ", repo_version)

    print("\nloaded modules:")

    modlist = ['scipy', 'numpy', 'sklearn', 'matplotlib',
               'binet', 'biutils', 'pandas', 'tensorflow', 'theano']
    modlist.extend(additional_modules)
    mod = [sys.modules[m] for m in modlist if m in sys.modules]
    mod.sort(key=lambda x: x.__name__)
    for m in mod:
        try:
            print("\t", m.__name__, m.__version__)
        except AttributeError:
            pass


def get_timestamp(fmt='%y%m%d_%H%M'):
    '''Returns a string that contains the current date and time.

    Suggested formats:
        short_format=%y%m%d_%H%M  (default)
        long format=%Y%m%d_%H%M%S
    '''
    import datetime
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, fmt)


def get_experiment_id():
    '''Returns a string that should be unique for each runs of an experiment.

    Sometimes, using a timestamp is not enough, e.g. when we start multiple
    threads at the same time. For those rare occasions, we use a short
    timestamp and the PID.
    '''

    ts = get_timestamp('%y%m%d%H%M')
    pid = os.getpid()
    return "%s_%d" % (ts, pid)


def heuristic_svm_c(x):
    ''' Heuristic for setting C for linear SVMs by Thorsten Joachims.'''
    c = 0
    n = x.shape[0]
    for i in range(n):
        c += np.sqrt(x[i, ].dot(x[i, ]))
    c /= n
    return 1.0 / c


def calculate_confusion_matrix(y_true, y_pred, n_classes=None, labels=None):
    """ Calculates a confusion matrix.

    Note: this is much faster than sklearn.metrics.confusion_matrix, but
          should be otherwise equivalent.

    Inputs:
        y_true: groundtruth labels
        y_pred: predictions
        labels: labels to consider (optional, if not given, all labels
                will be considered)

    Returns:
        Confusion matrix with columns=predicted classes, rows=true classes
    """
    a = y_true.astype(np.int64)
    b = y_pred.astype(np.int64)
    if n_classes is None and labels is None:
        raise RuntimeError("pass either number of classes or allowed labels")
    n = n_classes
    if labels is not None:
        n = labels.max() + 1  # largest label value
        idx = np.in1d(y_true, labels)
        a = a[idx]
        b = b[idx]
    return np.bincount(n * a + b, minlength=n**2).reshape(n, n)


def random_seed():
    '''
    Returns a seed that should be different for each process.
    This is useful if we start many processes at the same time.
    '''
    import time
    return np.uint32(hash(os.getpid() + time.time()) % 4294967295)


def linear_decay(t, v_start, v_end, n_steps):
    ''' Linearly decays learning rate from v_start to v_end in t_end timesteps '''
    if t >= n_steps:
        return v_end
    return v_start + t*(v_end - v_start) / n_steps


def exponential_decay(t, v_start, v_end, n_steps):
    ''' Exponentially decays a value (e.g. learning rate) from v_start to v_end in t_end timesteps '''
    if t >= n_steps:
        return v_end
    f = np.power(v_end / v_start, 1/(n_steps-1))
    return v_start * (f**t)


def startup_bookkeeping(logdir, curent_file):
    '''
    Performs common operations at start of each run.

    Writes PID and argv into files in the logdir and copies all the source
    files in the from current directory there as well.

    logdir is expected to be a pathlib.Path
    '''

    import shutil
    from pathlib import Path

    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)

    with open(logdir / "pid", 'w') as f:  # write PID so we can abort from outside
        f.write(str(os.getpid()))
        f.write("\n")

    # make copy of current source files
    dest = logdir / 'sources'
    dest.mkdir(exist_ok=True)
    localdir = Path(curent_file).parent
    pyfiles = localdir.glob("*.py")
    for fn in localdir.glob("*.py"):
        shutil.copy(fn, dest / fn)

    with open(logdir / "argv", 'w') as f:
        f.write(' '.join(sys.argv))
        f.write("\n")
