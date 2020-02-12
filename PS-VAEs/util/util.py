from __future__ import print_function
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageDraw
import inspect, re
import os
import collections
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

color = [[0,0,0],[255,0,0],[0,128,0],[0,255,0],[0,0,128],[0,0,255],[128,128,0],[128,255,0],
        [128,0,128],[128,0,255],[255,128,0],[255,255,0],[255,0,128],[255,0,255],[128,128,128],[128,128,255],
        [128,255,128],[128,255,255],[255,128,128],[255,255,128],[255,128,255],[255,255,255],[0,128,128],[0,128,255],
        [0,255,128],[0,255,255],[64,255,64],[255,64,64],[64,64,255],[64,64,0],[64,0,64],[128,0,0]]
joint_connection = [[1, 0], [2, 1], [3, 2],
                                 [2, 4], [2, 5], [4, 6], [5, 7], [6, 8], [7, 9], [8, 10], [9, 11],
                                 [3, 12], [3, 13], [12, 14], [13, 15], [14, 16], [15, 17]]
joint_colors = np.array([(63,0,0), (127,255,0), (191,255,191), (127,255,127),
                                      (63,127,0), (0,191,63), (127,63,0), (0,63,127), (255,63,255), (63,255,255), (255,63,0), (0,63,255),
                                      (63,0,63), (63,0,127), (255,127,127), (63,255,63), (191,127,63), (63,63,0)])
def onten(image_tensor):
    return image_tensor.detach().cpu().float().numpy()
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor,depth_px = None):
    if len(image_tensor.size()) == 3:
        #disc map or prob_num map
        image_numpy = image_tensor.cpu().float().numpy()
        return np.transpose(image_numpy, (1, 2, 0))
    image_numpy = image_tensor.cpu().float().numpy()
    # label image
    if image_numpy.shape[1] == 32:
        x, y = image_numpy.shape[-2:]
        image_numpy = np.argmax(image_numpy[0],axis=0).reshape((x*y))
        image_numpy = [list(color[pr]) for pr in image_numpy]
        image_numpy = np.resize(np.array(image_numpy),(x,y,3))
    # joint image
    elif image_numpy.shape[1] == 18:
        apply_gauss = [cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.3) for img in image_numpy[0]]
        joint_pos_2d = np.zeros((18,2))
        joint_pos_2d = np.array([np.unravel_index(np.argmax(joint), joint.shape) for joint in apply_gauss])
        if depth_px is None:
            depth_px = np.zeros((image_numpy.shape[-2],image_numpy.shape[-1],3))
        elif depth_px.size()[1] == 1:
            #remove mask
            depth_px = depth_px[0,0].cpu().float().numpy()
            depth_px = np.repeat(depth_px, 3).reshape((depth_px.shape[-2], depth_px.shape[-1], 3))
            depth_px = np.array((depth_px + 1) * 128, dtype=np.uint8)
        else:
            x, y = depth_px.shape[-2:]
            depth_px = np.argmax(depth_px[0].cpu().float().numpy(),axis=0).reshape((x*y))
            depth_px = [list(color[pr]) for pr in depth_px]
            depth_px = np.resize(np.array(depth_px, dtype=np.uint8),(x,y,3))

        img = Image.fromarray(depth_px)
        draw = ImageDraw.Draw(img)
        for p, c in joint_connection:
            if joint_pos_2d[p][1] is None or joint_pos_2d[p][0] is None or joint_pos_2d[c][1] is None or joint_pos_2d[c][0] is None:
                continue
            if True:
            #if p in visible_joints and c in visible_joints:
                draw.line((joint_pos_2d[p][1], joint_pos_2d[p][0], joint_pos_2d[c][1], joint_pos_2d[c][0]),
                          fill=tuple(joint_colors[c, :]),width=5)
        image_numpy = np.array(img)
    #RGB image
    elif image_numpy.shape[1] == 3:
        image_numpy = np.array((np.transpose(image_numpy[0], (1, 2, 0)) + 1) * 127.5,dtype=np.uint8)
    else:
        image_numpy = (np.transpose(image_numpy[0,-1:], (1, 2, 0)) + 1) * 2047.5
    return image_numpy

def gkern_2d(size=5, sigma=3, channel=1):
    # Create 2D gaussian kernel
    dirac = np.zeros((size, size))
    dirac[size//2, size//2] = 1
    mask = gaussian_filter(dirac, sigma)
    # Adjust dimensions for torch conv2d
    return np.stack([np.expand_dims(mask, axis=0)] * channel)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_image(image_numpy, image_path, heatmap = None):
    if np.max(image_numpy) - np.min(image_numpy) < 10:
        #discriminator output
        if len(image_numpy.shape) == 3:
            image_numpy = image_numpy[:,:,0]
        plt.imshow(image_numpy,vmax = 1, vmin = -1, cmap=matplotlib.cm.jet)
        plt.savefig(image_path)
        np.save(image_path.replace("png","npy"),image_numpy)
        return
    if len(image_numpy.shape) == 2:
        image_numpy = image_numpy[:,:,np.newaxis]
    if image_numpy.shape[-1] == 3 and np.max(image_numpy) < 256:
        #label image or joint image or RGB image
        plt.imshow(image_numpy)
    else:
        plt.imshow(image_numpy[:,:,-1],vmax=4096,vmin=0,cmap=matplotlib.cm.jet)
    plt.savefig(image_path)
    if heatmap is None:
        np.save(image_path.replace("png","npy"),image_numpy)
    else:
        np.save(image_path.replace("png","npy"),heatmap)
def save_error(error, path):
    #avg = np.mean(error, axis=0)
    np.save(os.path.join(path,"error.npy"), error)
    #print(avg)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
