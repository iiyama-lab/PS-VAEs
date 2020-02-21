import glob
import numpy as np
import cv2
import sys
import os
import gzip
import operator
import struct
from functools import reduce
from urllib.parse import urljoin
from scipy.io import loadmat
import logging
import requests

#We referenced https://github.com/erictzeng/adda

logger = logging.getLogger(__name__)

def maybe_download(url, dest):
    """Download the url to dest if necessary, optionally checking file
    integrity.
    """
    if not os.path.exists(dest):
        logger.info('Downloading %s to %s', url, dest)
        download(url, dest)

def download(url, dest):
    """Download the url to dest, overwriting dest if it already exists."""
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

class DatasetGroup(object):

    def __init__(self, per, name, path=None, download=True):
        self.name = name      
        self.percentage = per
        if path is None:
            path = os.getcwd()
           #             path = os.path.join(os.getcwd(), 'data')
        self.path = path
        if download:
            self.download()

    def get_path(self, *args):
        return os.path.join(self.path, self.name, *args)

    def download(self):
        """Download the dataset(s).
        This method only performs the download if necessary. If the dataset
        already resides on disk, it is a no-op.
        """
        pass

    def make_dataset(self, train_images, train_labels, test_images, test_labels):
        # for data augumentation in usps
        lis = [0,1,-1,2,-2,3,-3,4,-4]
        
        #amount of each digit
        NumOfOne = int(4500 * self.percentage / (100 - self.percentage))
        NumOfElse = 500
        
        # to reproduce the paper. this should be removed.
        imbalance_data_list = [[4000,4000],[4500,2000],[5400,1400],[6000,1000],[6300,700]]
        if self.name == "mnist" or self.name == "svhn":
            NumOfOne = imbalance_data_list[self.percentage // 10 - 1][0]
            NumOfOne = imbalance_data_list[self.percentage // 10 - 1][1]

        if not os.path.exists("train_"+self.name+"_"+str(self.percentage)):
            os.mkdir("train_"+self.name+"_"+str(self.percentage))
            for num in range(10):
                count = 0
                a = -1
                flag = True
                while flag: #Exit when getting enough data
                    a += 1
                    for i, (img, lab) in enumerate(zip(train_images,train_labels)):
                        if lab == 10: lab = 0 #SVHN use 10 as 0
                        if num != lab:
                            continue
                        if (num != 1 and count >= NumOfElse) or (num == 1 and count >= NumOfOne):
                            flag = False
                            break
                        M = np.float32([[1,0,lis[a]],[0,1,0]])
                        img = cv2.resize(img,(32,32),interpolation=cv2.INTER_NEAREST)
                        img = cv2.warpAffine(img,M,(32,32)) #data augumentation in usps
                        if len(img.shape) == 2: img = img[:,:,None]  
                        label = np.full((32,32,1), lab)
                        img = np.concatenate([img,label],2) #last channel shows groundtruth digit
                        img = np.transpose(img,(2,0,1))  # this order is suitable for training data in pytorch
                        np.save(os.path.join("train_"+self.name+"_"+str(self.percentage),str(lab)+"_"+str(count)+".npy"), img)
                        count += 1
        if not os.path.exists("test_"+self.name):
            os.mkdir("test_"+self.name)
            for i, (img, lab) in enumerate(zip(test_images,test_labels)):
                if lab == 10:lab = 0  #SVHN use 10 as 0
                img = cv2.resize(img,(32,32),interpolation=cv2.INTER_NEAREST)
                if len(img.shape) == 2: img = img[:,:,None]  #mnist have no channel
                label = np.full((32,32,1), lab)
                img = np.concatenate([img,label],2) #last channel shows groundtruth digit
                img = np.transpose(img,(2,0,1)) # this order is suitable for training data in pytorch
                np.save(os.path.join("test_"+self.name,str(lab)+"_"+str(i)+".npy"), img)


class MNIST(DatasetGroup):
    """The MNIST database of handwritten digits.
    Homepage: http://yann.lecun.com/exdb/mnist/
    Images are 28x28 grayscale images in the range [0, 1].
    """

    base_url = 'http://yann.lecun.com/exdb/mnist/'

    data_files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz',
            }

    num_classes = 10

    def __init__(self, par, path=None, shuffle=True):
        DatasetGroup.__init__(self, par, 'mnist', path)
        self.image_shape = (28, 28, 1)
        self.label_shape = ()
        self.shuffle = shuffle
        self._load_datasets()

    def download(self):
        data_dir = self.get_path()
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                maybe_download(url, path)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_images = self._read_images(abspaths['train_images'])
        train_labels = self._read_labels(abspaths['train_labels'])
        test_images = self._read_images(abspaths['test_images'])
        test_labels = self._read_labels(abspaths['test_labels'])
        self.make_dataset(train_images, train_labels, test_images, test_labels)

    def _read_datafile(self, path, expected_dims):
        #"""Helper function to read a file in IDX format."""
        base_magic_num = 2048
        with gzip.GzipFile(path) as f:
            magic_num = struct.unpack('>I', f.read(4))[0]
            expected_magic_num = base_magic_num + expected_dims
            if magic_num != expected_magic_num:
                raise ValueError('Incorrect MNIST magic number (expected '
                                 '{}, got {})'
                                 .format(expected_magic_num, magic_num))
            dims = struct.unpack('>' + 'I' * expected_dims,
                                 f.read(4 * expected_dims))
            buf = f.read(reduce(operator.mul, dims))
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(*dims)
            return data

    def _read_images(self, path):
        #"""Read an MNIST image file."""
        return (self._read_datafile(path, 3)
                .astype(np.float32)
                .reshape(-1, 28, 28, 1))

    def _read_labels(self, path):
        #"""Read an MNIST label file."""
        return self._read_datafile(path, 1)

class USPS(DatasetGroup):
    #"""USPS handwritten digits.
    #Homepage: http://statweb.stanford.edu/~hastie/ElemStatLearn/data.html
    #Images are 16x16 grayscale images in the range [0, 1].
    #"""

    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

    data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }

    num_classes = 10

    def __init__(self, par, path=None, shuffle=True, download=True):
        DatasetGroup.__init__(self, par, 'usps', path=path, download=download)
        self.image_shape = (16, 16, 1)
        self.label_shape = ()
        self.shuffle = shuffle
        self._load_datasets()

    def download(self):
        data_dir = self.get_path()
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                maybe_download(url, path)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_images, train_labels = self._read_datafile(abspaths['train'])
        test_images, test_labels = self._read_datafile(abspaths['test'])
        self.make_dataset(train_images, train_labels, test_images, test_labels)

    def _read_datafile(self, path):
        #"""Read the proprietary USPS digits data file."""
        labels, images = [], []
        with gzip.GzipFile(path) as f:
            for line in f:
                vals = line.strip().split()
                labels.append(float(vals[0]))
                images.append([float(val) for val in vals[1:]])
        labels = np.array(labels, dtype=np.int32)
        images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
        images = (images + 1) / 2
        return images, labels
class SVHN(DatasetGroup):
    """The Street View House Numbers Dataset.
    This DatasetGroup corresponds to format 2, which consists of center-cropped
    digits.
    Homepage: http://ufldl.stanford.edu/housenumbers/
    Images are 32x32 RGB images in the range [0, 1].
    """

    base_url = 'http://ufldl.stanford.edu/housenumbers/'

    data_files = {
            'train': 'train_32x32.mat',
            'test': 'test_32x32.mat',
            #'extra': 'extra_32x32.mat',
            }

    def __init__(self, par, path=None, shuffle=True):
        DatasetGroup.__init__(self, par, 'svhn', path=path)
        self.image_shape = (32, 32, 3)
        self.label_shape = ()
        self.shuffle = shuffle
        self._load_datasets()

    def download(self):
        data_dir = self.get_path()
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                maybe_download(url, path)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_mat = loadmat(abspaths['train'])
        train_images = train_mat['X'].transpose((3, 0, 1, 2))
        train_labels = train_mat['y'].squeeze()
        train_images = train_images.astype(np.float32)
        test_mat = loadmat(abspaths['test'])
        test_images = test_mat['X'].transpose((3, 0, 1, 2))
        test_images = test_images.astype(np.float32)
        test_labels = test_mat['y'].squeeze()
        self.make_dataset(train_images, train_labels, test_images, test_labels)

if __name__ == '__main__':
    #argv[1]:dataset
    #argv[2]:percentage
    par = int(sys.argv[2])
    if sys.argv[1] == "mnist":
        MNIST(par)
    if sys.argv[1] == "usps":
        USPS(par)
    if sys.argv[1] == "svhn":
        SVHN(par)
