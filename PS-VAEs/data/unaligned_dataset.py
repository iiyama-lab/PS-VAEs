import os.path, glob
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch
import cv2
import re

class UnalignedDataset(BaseDataset):
    def __init__(self, opt, val=False):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.val = val
        self.transform = get_transform(opt)
        source = opt.source
        target = opt.target
        root = opt.dataroot + '/' + opt.phase
        if opt.est_mnist and val:
            target = target.replace('_imb','')
            target = re.sub(r'[0-9]wari','',target)
        elif opt.est_mnist and not opt.isTrain:
            target = target.replace('_imb','')
            target = re.sub(r'[0-9]wari','',target)
        elif val:
            root = root.replace('train','test')

        self.dirs = [os.path.join(root +'_'+ target), os.path.join(opt.dataroot, opt.phase +'_'+ source)]

        self.paths = [sorted(make_dataset(d,40000,opt.isTrain)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

    def load_image(self, dom, idx):
        cv=0
        path = self.paths[dom][idx]
        #img = Image.open(path).convert('RGB')
        img = np.array(np.load(path),dtype=np.float)
        if self.opt.est_mnist:
            if img.shape[0] == 4:
                #svhn
                img[:3,:,:] = np.clip(img[:3,:,:] / 128 - 1, -1, 1)
                if img[-1,0,0] == 10:
                    img[-1] = 0
                elif self.opt.input_nc == 1 and self.opt.output_nc == 1:
                    img = np.stack([(img[0]+ img[1] +  img[2])/3, img[-1]])
            else:
                if np.max(img[0]) <= 1:
                    #usps
                    img = np.stack([cv2.resize(i,(32,32),interpolation=cv2.INTER_NEAREST) for i in img])
                    img[:1] = img[:1] * 2 - 1
                else:
                    #mnist
                    img[:1,:,:] = np.clip(img[:1,:,:] / 128 - 1, -1, 1)
                if self.opt.ufdn:
                    dep = img[0]
                    p = random.uniform(0, 1)
                    if p < 0.5:
                        dep = -dep
                    img = np.stack([dep, dep, dep, img[-1]])
                elif self.opt.input_nc == 3 and self.opt.output_nc == 3:
                    img = np.stack([img[0],img[0],img[0],img[-1]])

            img = torch.from_numpy(img)
            return img, path, 32, 32

        if len(img.shape) == 2:
                img = img[np.newaxis,:,:]
        x,y = img.shape[1],img.shape[2]
        img[0,:,:] = np.clip(img[0,:,:] / 2048 - 1, -1, 1)
        img = torch.from_numpy(img)

        return img, path, x, y

    def __getitem__(self, index):
        if not self.opt.isTrain and not self.val:
            # Choice of domain order sequential during test
            DA = index % len(self.dirs)
            DA = 0
            # Choice of image within domain not always ordered
            if self.val:
                index_A = random.randint(0, self.sizes[DA] - 1)
            elif self.opt.serial_test:
                index_A = index // len(self.dirs)
                if not self.opt.est_mnist:
                    pitch = 500 // self.opt.how_many
                    index_A = index * pitch % self.sizes[DA]
            else:
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            # Choose two of our domains to perform a pass on
            #DA, DB = random.sample(range(len(self.dirs)), 2)

            #DA : 0 : rea  DB : 1 : syn
            DA, DB = 0, 1
            index_A = random.randint(0, self.sizes[DA] - 1)

        A_img, A_path, ax, ay = self.load_image(DA, index_A)
        bundle = {'A': A_img, 'DA': DA, 'path': A_path, 'ax': ax, 'ay': ay}

        if self.opt.isTrain and not self.val:
            index_B = random.randint(0, self.sizes[DB] - 1)
            B_img, _, bx, by = self.load_image(DB, index_B)
            bundle.update( {'B': B_img, 'DB': DB, 'bx': bx, 'by': by} )

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'UnalignedDataset'
