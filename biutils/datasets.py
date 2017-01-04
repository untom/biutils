#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Dataset handling within buitils.

Copyright © 2013-2016 Thomas Unterthiner.
Licensed under GPL, version 2 or a later (see LICENSE.rst)

buitils stores datasets as HDF5 files. A dataset is comprised of 6 matrices:
x_tr, y_tr, x_va, y_va, x_te, y_te
'''

from __future__ import absolute_import, division, print_function

import os
import platform
import logging
import numpy as np
from biutils.misc import download_file

try:
    import h5py
except ImportError:
    import warnings
    warnings.warn("h5py unavailable")

# some machine specific paths for bioinf@jku machines
__datadir = {'tomlap': '/media/scratch/data',
             'blucomp': '/media/scratch/data'}
_DATA_DIRECTORY = __datadir.get(platform.node(), os.path.expanduser("~/data"))


def load_dataset(dataset_name, return_testset=False,
                 dtype=np.float32, revert_scaling=False):
    '''Loads a dataset, given the filename of the HDF5 file.

    Returns 4 tuple of X, y, x_vaid, y_vaid)
    '''
    if not dataset_name.endswith(".h5"):
        fname = os.path.join(_DATA_DIRECTORY, dataset_name + ".h5")
    else:
        fname = os.path.join(_DATA_DIRECTORY, dataset_name)

    # try to create standard datset if it doesn't exist yet
    if not os.path.exists(fname):
        createfuncs = {
            'mnist': _create_mnist,
            'norb': _create_norb,
            'cifar10': _create_cifar10_flat,
            'cifar10_img': _create_cifar10_img,
            'mnist_basic': _create_mnist_basic,
            'mnist_bgimg': _create_mnist_bgimg,
            'mnist_bgrand': _create_mnist_bgrand,
            'mnist_rot': _create_mnist_rot,
            'rectangles': _create_rectangles,
            'convex': _create_convex,
            'covertype': _create_covertype,
            'enwik8': _create_enwik8,
            'tox21': _create_tox21}
        cf = createfuncs.get(dataset_name, None)
        if cf is not None:
            l = logging.getLogger(__name__)
            l.warning("%s does not exist, trying to create it" % fname)
            cf(_DATA_DIRECTORY)

    if not os.path.exists(fname):
        raise RuntimeError("File %s does not exist" % fname)
    with h5py.File(fname) as dataset:
        if dataset_name == "enwik8":
            ds_keys = ['train', 'valid', 'test']
        else:
            ds_keys = ['x_train', 'y_train', 'x_valid', 'y_valid']
            if return_testset:
                ds_keys.extend(['x_test', 'y_test'])

        data = []
        s = dataset['scale'][:] if 'scale' in dataset else 1.0
        c = dataset['center'][:] if 'center' in dataset else 0.0
        for k in ds_keys:
            if k.endswith('x') and revert_scaling:
                data.append(((dataset[k][:] * s)+c).astype(dtype))
            else:
                data.append(dataset[k][:].astype(dtype))
    import gc
    gc.collect()
    return data


def _to_one_hot_encoding(labels, dtype=np.float64):
    labels = labels.reshape((labels.shape[0], 1))
    '''Creates a one-hot encoding of the labels.'''
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(dtype=dtype)
    return enc.fit_transform(labels).toarray()


def _shuffle(data, labels):
    ''' Shuffles the data and the labels.'''
    np.random.seed(42)  # Make sure the same file is produced on each machine
    idx = np.array(range(data.shape[0]))
    np.random.shuffle(idx)
    data = data[idx, :]
    labels = labels[idx, :]
    return data, labels, idx


def _read_mnist_image(filename):
    import struct
    import gzip
    with gzip.open(filename) as f:
        buf = f.read(16)
        magic, n_items, xsize, ysize = struct.unpack(">iiii", buf)
        assert magic == 2051  # magic number
        n_features = xsize*ysize
        data = np.zeros((n_items, n_features), dtype=np.uint8)
        for i in range(n_items):
            buf = f.read(n_features)
            x = struct.unpack("B"*n_features, buf)
            data[i, :] = x
    return data


def _read_mnist_label(filename):
    import struct
    import gzip
    with gzip.open(filename) as f:
        buf = f.read(8)
        magic, n_items = struct.unpack(">ii", buf)
        assert magic == 2049  # magic number
        data = np.zeros(n_items, dtype=np.uint8)
        buf = f.read(n_items)
        data[:] = struct.unpack("B"*n_items, buf)
    return data.reshape(-1, 1)


def _read_norb_data(filename):
    import struct
    import gzip
    with gzip.open(filename) as f:
        buf = f.read(8)
        magic, ndims = struct.unpack("<ii", buf)
        if magic == 0x1e3d4c55:
            dt = np.dtype(np.uint8)
        elif magic == 0x1e3d4c54:
            dt = np.dtype(np.uint32)
        else:
            assert(False)
        n = max(ndims, 3)
        buf = f.read(n * 4)
        dims = struct.unpack('<' + ('i'*n), buf)
        nitems = dims[0]
        nfeatures = int(np.prod(dims[1:]))
        data = np.empty((nitems, nfeatures), dtype=dt.type)

        # we have to iterate here, as doing it all at once
        # might cause a MemoryError
        for i in range(nitems):
            buf = f.read(nfeatures*dt.itemsize)
            data[i] = struct.unpack(dt.char*nfeatures, buf)
    return data


def _store(data, filename, other=None):
    #
    # Note: deactivating compression got a MASSIVE boost in read-speed.
    # Our only compression-choice was gzip, as rhdf5 (R implementation)
    # could not handle LZO.
    # without compression, CIFAR10 can be read in <1 second in R
    # (filesize ~750MB)
    # with GZIP, no matter what compression level, the times were ~40s.
    # (even though GZIP with compression_opts = 0 resulted in a file of 750MB)
    # (compression_opts = 9 reached ~250 MB)
    #
    logging.getLogger(__name__).info("saving into %s ..." % filename)
    with h5py.File(filename, "w") as f:
        for i in range(len(data)):
            f.create_dataset('x_' + data[i][0], data=data[i][1])
            f.create_dataset('y_' + data[i][0], data=data[i][2])#, compression="gzip", compression_opts = 0)
        if other:
            for k in other:
                f.create_dataset(k, data=other[k])


def _process_and_store(data, filename, other=None, dtype=np.float32):
    '''Shuffles, converts and stores the data.

    Shuffles training and testset, converts the data to np.float64 and stores
    it. `other` can be dictionary of additional data to store.

    data is expected to be a list of datasets, where each dataset is a list of
    [name, data, labels]. I.e. a normal train/testset split would be
    data = [ ['train', traindata, trainlabels], ['test', testdata, testlabels]]
    '''
    logger = logging.getLogger(__name__)
    logger.info("shuffling...")
    for i in range(len(data)):
        data[i][1], data[i][2], _ = _shuffle(data[i][1], data[i][2])
    logger.info("converting...")
    for i in range(len(data)):
        data[i][1] = data[i][1].astype(dtype)
        data[i][2] = _to_one_hot_encoding(data[i][2], dtype=dtype)

    # scale to [0, 1] based on training data
    s = data[0][1].max()
    for i in range(len(data)):
        data[i][1] /= s
    if other is None:
        other = {}
    other['scale'] = s*np.ones(data[0][1].shape[1])
    _store(data, filename, other)


def _split_dataset(data, labels, fraction):
    """ Splits a dataset into two set, with the first part
        obtaining fraction % of the data."""
    n = int(data.shape[0] * fraction + 0.5)
    idx = np.random.choice(range(data.shape[0]), n, replace=False)
    return (data[idx, ], labels[idx],
            np.delete(data, idx, 0), np.delete(labels, idx, 0))


def _create_mnist(download_dir):
    ''' MNIST dataset from yann.lecun.com/exdb/mnist/  '''
    from os.path import join
    logger = logging.getLogger(__name__)
    logger.info("reading data...")
    urlbase = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    destdir = join(download_dir, "raw")
    for fname in files:
        download_file(urlbase, destdir, fname)
    x_tr = _read_mnist_image(join(destdir, "train-images-idx3-ubyte.gz"))
    y_tr = _read_mnist_label(join(destdir, "train-labels-idx1-ubyte.gz"))
    x_te = _read_mnist_image(join(destdir, "t10k-images-idx3-ubyte.gz"))
    y_te = _read_mnist_label(join(destdir, "t10k-labels-idx1-ubyte.gz"))

    x_tr, y_tr, x_va, y_va = _split_dataset(x_tr, y_tr, 5/6.0)
    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    _process_and_store(data, join(download_dir, "mnist.h5"))


def _create_norb(download_dir):
    '''Small NORB dataset from www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/ '''

    urlbase = "http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    dst = os.path.join(download_dir, "raw")
    x_tr = _read_norb_data(download_file(urlbase, dst,
        'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz'))
    y_tr = _read_norb_data(download_file(urlbase, dst,
        'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz'))
    i_tr = _read_norb_data(download_file(urlbase, dst,
        'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz'))
    x_te = _read_norb_data(download_file(urlbase, dst,
        'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz'))
    y_te = _read_norb_data(download_file(urlbase, dst,
        'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz'))

    # instead of assigning the validation set randomly, we pick one of the
    # "instances" of the training set. This is much better than doing
    # it randomly!
    fold = i_tr[:, 0].ravel()
    vi = (fold == 4)  # let's make instance 4 the validation-instance
    x_va, x_tr = x_tr[vi], x_tr[~vi]
    y_va, y_tr = y_tr[vi], y_tr[~vi]
    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    _process_and_store(data, os.path.join(download_dir, "norb.h5"))


def create_cifar10(download_dir=_DATA_DIRECTORY):
    logger = logging.getLogger(__name__)
    logger.info('reading CIFAR10 data...')
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    fname = download_file(url, os.path.join(download_dir, "raw"))
    import tarfile
    with tarfile.open(fname) as tf:
        filemembers = tf.getmembers()
        files = [f.name for f in filemembers if "data_batch" in f.name]
        files.sort()

        def _read_file(fn):
            f = tf.extractfile(fn)
            tmp = np.frombuffer(f.read(), np.uint8).reshape(-1, 3073)
            return tmp[:, 0].reshape(-1, 1), tmp[:, 1:].reshape(-1, 3*32*32)

        # save last batch as validation
        traindata = [_read_file(fn) for fn in files[0:len(files)-1]]
        y_tr = np.vstack([t[0] for t in traindata])
        x_tr = np.vstack([t[1] for t in traindata])

        y_va, x_va = _read_file(files[-1])
        y_te, x_te = _read_file('cifar-10-batches-bin/test_batch.bin')
        return x_tr, y_tr.ravel(), x_va, y_va.ravel(), x_te, y_te.ravel()


def _create_cifar10_flat(download_dir):
    ''' CIFAR-10, from www.cs.toronto.edu/~kriz/cifar.html.'''
    x_tr, y_tr, x_va, y_va, x_te, y_te = create_cifar10(download_dir)

    data = [['train', x_tr, y_tr.reshape(-1, 1)],
            ['valid', x_va, y_va.reshape(-1, 1)],
            ['test', x_te, y_te.reshape(-1, 1)]]
    dst = os.path.join(download_dir, "cifar10.h5")
    _process_and_store(data, dst)
    # imshow(np.rot90(traindata[882, ].reshape((3, 32, 32)).T), origin="lower")


def _create_cifar10_img(download_dir):
    ''' CIFAR-10 in nbatches x width x height x channels format
    from www.cs.toronto.edu/~kriz/cifar.html.'''
    x_tr, y_tr, x_va, y_va, x_te, y_te = create_cifar10(download_dir)
    x_tr, x_va, x_te = [x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                        for x in (x_tr, x_va, x_te)]

    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    dst = os.path.join(download_dir, "cifar10_img.h5")
    _store(data, dst)


def _handle_larochelle_icml2007(download_dir, fn, train_data_file,
                                test_data_file, rotate_images=True):
    '''Basic procedure to load the datasets from Larochelle et al., ICML 2007.
    Unfortunately the structure of the datasets differs sometimes,
    so we need this abstraction.

    fn = name of the zip file (w/o extension)
    train_data_file: name of the training set file within the archive
    test_data_file: name of the test set file within the archive
    rotate_images: rotate images (needed if file is in column-major format)
    '''
    import zipfile
    urlbase = "http://www.iro.umontreal.ca/~lisa/icml2007data/"
    dst = os.path.join(download_dir, "raw")
    f = download_file(urlbase, dst, '%s.zip' % fn)
    with zipfile.ZipFile(f) as zf:
        tmp = np.loadtxt(zf.open(train_data_file))
        x_tr, y_tr = tmp[:, :-1].copy(), tmp[:, -1].copy()
        tmp = np.loadtxt(zf.open(test_data_file))
        x_te, y_te = tmp[:, :-1].copy(), tmp[:, -1].copy()
        y_tr = y_tr.reshape((-1, 1))
        y_te = y_te.reshape((-1, 1))
        if rotate_images:
            n = int(np.sqrt(x_tr.shape[1]))
            x_tr = np.rollaxis(x_tr.reshape(x_tr.shape[0], n, n), 2, 1)
            x_tr = x_tr.reshape(-1, n*n)
            x_te = np.rollaxis(x_te.reshape(x_te.shape[0], n, n), 2, 1)
            x_te = x_te.reshape(-1, n*n)
        return x_tr, y_tr, x_te, y_te


def _create_mnist_basic(download_dir):
    x_tr, y_tr, x_te, y_te = _handle_larochelle_icml2007(download_dir, 'mnist',
        'mnist_train.amat', 'mnist_test.amat', rotate_images=False)
    x_tr, y_tr, x_va, y_va = _split_dataset(x_tr, y_tr, 5/6.0)
    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    _process_and_store(data, os.path.join(download_dir, "mnist_basic.h5"))


def _create_mnist_bgimg(download_dir):
    x_tr, y_tr, x_te, y_te = _handle_larochelle_icml2007(download_dir,
        'mnist_background_images',
        'mnist_background_images_train.amat',
        'mnist_background_images_test.amat')
    x_tr, y_tr, x_va, y_va = _split_dataset(x_tr, y_tr, 5/6.0)
    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    _process_and_store(data, os.path.join(download_dir, "mnist_bgimg.h5"))


def _create_mnist_bgrand(download_dir):
    x_tr, y_tr, x_te, y_te = _handle_larochelle_icml2007(download_dir,
        'mnist_background_random',
        'mnist_background_random_train.amat',
        'mnist_background_random_test.amat')
    x_tr, y_tr, x_va, y_va = _split_dataset(x_tr, y_tr, 5/6.0)
    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    _process_and_store(data, os.path.join(download_dir, "mnist_bgrand.h5"))


def _create_mnist_rot(download_dir):
    x_tr, y_tr, x_te, y_te = _handle_larochelle_icml2007(download_dir,
        'mnist_rotation_new',
        'mnist_all_rotation_normalized_float_train_valid.amat',
        'mnist_all_rotation_normalized_float_test.amat')
    x_tr, y_tr, x_va, y_va = _split_dataset(x_tr, y_tr, 5/6.0)
    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    _process_and_store(data, os.path.join(download_dir, "mnist_rot.h5"))


def _create_rectangles(download_dir):
    x_tr, y_tr, x_te, y_te = _handle_larochelle_icml2007(download_dir,
        'rectangles', 'rectangles_train.amat', 'rectangles_test.amat')
    x_tr, y_tr, x_va, y_va = _split_dataset(x_tr, y_tr, 5/6.0)
    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    _process_and_store(data, os.path.join(download_dir, "rectangles.h5"))


def _create_convex(download_dir):
    x_tr, y_tr, x_te, y_te = _handle_larochelle_icml2007(download_dir,
        'convex', 'convex_train.amat', '50k/convex_test.amat')
    x_tr, y_tr, x_va, y_va = _split_dataset(x_tr, y_tr, 5/6.0)
    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    _process_and_store(data, os.path.join(download_dir, "convex.h5"))


def _create_covertype(download_dir):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/'
    destdir = os.path.join(download_dir, "raw")
    fn = download_file(url, destdir, 'covtype.data.gz')
    import gzip
    import pandas as pd
    with gzip.open(fn, "rb") as gzfile:
        x = pd.read_csv(gzfile, header=None).values

    x, y = x[:, :-1].astype(np.float64), x[:, -1]
    y -= 1  # make classes 0-based

    # split into test- and validationset
    from sklearn.cross_validation import train_test_split
    x, x_te, y, y_te = train_test_split(x, y, test_size=0.1)  # create testset
    x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.25)

    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    y_va = lb.transform(y_va)
    y_te = lb.transform(y_te)

    # Most values are binary, except for these, so let's standardize them
    quant_idx = [0, 1, 2, 3, 4, 5, 9]  # real numbers
    int_idx = [6, 7, 8]                # integers from [0, 255)
    idx = quant_idx + int_idx
    from sklearn.preprocessing import StandardScaler as Scaler
    scaler = Scaler()
    x_tr[:, idx] = scaler.fit_transform(x_tr[:, idx])
    x_va[:, idx] = scaler.transform(x_va[:, idx])
    x_te[:, idx] = scaler.transform(x_te[:, idx])
    data = [['train', x_tr,  y_tr],
            ['valid', x_va,  y_va],
            ['test', x_te,  y_te]]
    m = np.zeros(x_tr.shape[1])
    m[quant_idx+int_idx] = scaler.mean_
    s = np.ones(x_tr.shape[1])
    s[quant_idx+int_idx] = scaler.std_
    other = {'center': m, "scale": s}
    _store(data, os.path.join(download_dir, "covertype.h5"), other)


def _create_enwik8(download_dir):
    import pandas as pd
    '''Prepares the enwik8/hutter prize data: an extract from wikipedia.'''
    urlbase = 'http://mattmahoney.net/dc/'
    destdir = os.path.join(download_dir, "raw")
    fn = download_file(urlbase, destdir, 'enwik8.zip')

    # we first read the text as UTF-8, and then map each present character
    # to a number, instead of using UTF-8 bytes directly
    import zipfile
    with zipfile.ZipFile(fn, "r") as zf:
        with zf.open("enwik8") as z:
            text_train = z.read(96*10**6).decode("utf8")
            text_valid = z.read(2*10**6).decode("utf8")
            text_test = z.read(2*10**6).decode("utf8")
            assert(len(z.read()) == 0)  # make sure we read everything

    # ignore "uncommon" characters.
    # In "Generating Sequences With Recurrent Neural Networks"
    # Alex Graves says that there are 205 distinct single-byte characters.
    # However the following will only yield 196. No idea where Alex
    # got the rest of them ?-)
    dt = np.uint8
    data_tr = np.array([ord(c) for c in text_train if ord(c) < 256], dtype=dt)
    data_va = np.array([ord(c) for c in text_valid if ord(c) < 256], dtype=dt)
    data_te = np.array([ord(c) for c in text_test if ord(c) < 256], dtype=dt)
    cnt = pd.value_counts(data_tr)

    del(text_train, text_valid, text_test)
    import gc
    gc.collect()

    # remove characters with <=10 occourences (there are 16 of those)
    # (we use a lookup table, othewise it takes forever)
    count_loopup = np.zeros(256, np.int64)
    count_loopup[cnt.index.values] = cnt.values
    occ = count_loopup[data_tr]
    data_tr = data_tr[occ > 10]
    data_va = data_va[count_loopup[data_va] > 10]
    data_te = data_te[count_loopup[data_te] > 10]

    decode_lookup = 255 * np.ones(256, np.uint8)
    u = np.unique(data_tr)
    decode_lookup[:len(u)] = u
    encode_lookup = np.iinfo(np.uint16).max * np.ones(256, np.uint16)
    for c, e in enumerate(u):
        encode_lookup[e] = c
    code_tr = encode_lookup[data_tr]
    code_va = encode_lookup[data_va]
    code_te = encode_lookup[data_te]
    assert(np.all(decode_lookup[code_tr] == data_tr))
    assert(np.all(code_tr <= 255))
    assert(np.all(code_va <= 255))
    assert(np.all(code_te <= 255))
    del(data_tr, data_va, data_te)
    gc.collect()

    fname = os.path.join(download_dir, "enwik8.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset('train', data=code_tr)
        f.create_dataset('valid', data=code_va)
        f.create_dataset('test', data=code_te)
        f.create_dataset('encode', data=encode_lookup)
        f.create_dataset('decode', data=decode_lookup)


def create_tox21(sparsity_cutoff, va_folds,
                 dtype=np.float32, download_dir=_DATA_DIRECTORY):
    ''' Creates a preprocessed version of the tox21 dataset.
    va_folds is a list of folds that are to be put into the validation set.
    '''
    from scipy import io
    import pandas as pd
    urlbase = "http://www.bioinf.jku.at/research/deeptox/"
    dst = os.path.join(download_dir, "raw")
    fn_x_tr_d = download_file(urlbase, dst, 'tox21_dense_train.csv.gz')
    fn_x_tr_s = download_file(urlbase, dst, 'tox21_sparse_train.mtx.gz')
    fn_y_tr = download_file(urlbase, dst, 'tox21_labels_train.csv.gz')
    fn_x_te_d = download_file(urlbase, dst, 'tox21_dense_test.csv.gz')
    fn_x_te_s = download_file(urlbase, dst, 'tox21_sparse_test.mtx.gz')
    fn_y_te = download_file(urlbase, dst, 'tox21_labels_test.csv.gz')
    cpd = download_file(urlbase, dst, 'tox21_compoundData.csv')

    y_tr = pd.read_csv(fn_y_tr, index_col=0)
    y_te = pd.read_csv(fn_y_te, index_col=0)
    x_tr_dense = pd.read_csv(fn_x_tr_d, index_col=0).values
    x_te_dense = pd.read_csv(fn_x_te_d, index_col=0).values
    x_tr_sparse = io.mmread(fn_x_tr_s).tocsc()
    x_te_sparse = io.mmread(fn_x_te_s).tocsc()

    # filter out very sparse features
    sparse_col_idx = ((x_tr_sparse > 0).mean(0) >= sparsity_cutoff).A.ravel()
    x_tr_sparse = x_tr_sparse[:, sparse_col_idx].A
    x_te_sparse = x_te_sparse[:, sparse_col_idx].A

    # filter out low-variance features
    dense_col_idx = np.where(x_tr_dense.var(0) > 1e-6)[0]
    x_tr_dense = x_tr_dense[:, dense_col_idx]
    x_te_dense = x_te_dense[:, dense_col_idx]

    # handle very large and exponential features
    # (Note experimentally, this doesn't seem to make a difference)
    xm = np.minimum(x_tr_dense.min(0), x_te_dense.min(0)) # avoid negative numbers
    log_x_tr = np.log10(x_tr_dense - xm+1e-8)
    log_x_te = np.log10(x_te_dense - xm+1e-8)
    exp_cols = np.where(x_tr_dense.ptp(0) > 10.0)
    x_tr_dense[:, exp_cols] = log_x_tr[:, exp_cols]
    x_te_dense[:, exp_cols] = log_x_te[:, exp_cols]


    # find the index of the validation items
    info = pd.read_csv(cpd, index_col=0)
    folds = info.CVfold[info.set != 'test'].values
    idx_va = np.zeros(folds.shape[0], np.bool)
    for fid in va_folds:
        idx_va |= (folds == float(fid))

    # normalize features
    from sklearn.preprocessing import StandardScaler, RobustScaler

    x_tr = np.hstack([x_tr_dense, x_tr_sparse])
    x_te = np.hstack([x_te_dense, x_te_sparse])

    s = RobustScaler()
    s.fit(x_tr[~idx_va])
    x_tr = s.transform(x_tr)
    x_te = s.transform(x_te)

    x_tr = np.tanh(x_tr)
    x_te = np.tanh(x_te)

    s = StandardScaler()
    s.fit(x_tr[~idx_va])
    x_tr = s.transform(x_tr)
    x_te = s.transform(x_te)

    return (x_tr[~idx_va].astype(dtype, order='C'),
            y_tr[~idx_va].values.astype(dtype, order='C'),
            x_tr[idx_va].astype(dtype, order='C'),
            y_tr[idx_va].values.astype(dtype, order='C'),
            x_te.astype(dtype, order='C'),
            y_te.values.astype(dtype, order='C'))


def _create_tox21(download_dir):
    sparsity_cutoff = 0.05
    validation_fold = [5, ]
    d = create_tox21(sparsity_cutoff, validation_fold)
    x_tr, y_tr, x_va, y_va, x_te, y_te = d
    data = [['train', x_tr,  y_tr],
            ['valid', x_va,  y_va],
            ['test',  x_te,  y_te]]
    _store(data, os.path.join(download_dir, "tox21.h5"))
