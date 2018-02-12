#!/usr/bin/env python
""" Test the validity of STAE. """
import os
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from cae import STAE

DATA_FOLDER = '/seagate_backup_plus_drive/Unsupvised_VAD/RawImgData/Avenue/Train'
VOL_LIST = ['vol{:02d}.mat'.format(n) for n in range(1, 17)]
VOL = os.path.join(DATA_FOLDER, VOL_LIST[0])
assert os.path.exists(VOL)

data = sio.loadmat(VOL)['vol']
#  height, width, num_frames = data.shape
# normalization
data = data.astype(np.float32, copy=False) / 255.
data_resize = np.zeros((data.shape[-1], 128, 128), dtype=np.float32)
for idx in range(data.shape[-1]):
    data_resize[idx] = cv2.resize(data[:, :, idx], (128, 128))
data = data_resize
#  data = data_resize.transpose(2, 0, 1)
# Reshape to [channels, depth, height, width]
#  data = np.expand_dims(data_resize.transpose(2, 0, 1), axis=0)

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        init.xavier_normal(m.weight.data)
        m.bias.data.zero_()


stae = STAE(1)
stae.apply(weights_init)
#  stae.cuda()


def train(epoch, batch_size, num_frames=16):
    """ Train the autoencoder for the given epoch. """
    num_volume = data.shape[0] // num_frames
    perm = np.random.permutation(num_volume)
    for idx in range(num_volume//batch_size):
        selected_idx = perm[idx*batch_size:(idx+1)*batch_size]
        batch = np.zeros((batch_size, 1, num_frames, 128, 128),
                         dtype=np.float32)
        for k, v in enumerate(selected_idx):
            batch[k] = data[v:v+num_frames]
        batch = torch.FloatTensor(batch)
        target = Variable(batch, volatile=True)
        batch = Variable(batch)
        recon, pred = stae(batch)
        print('Working on {}/{} iteration...'.format(idx+1, num_volume//batch_size))


if __name__ == '__main__':
    train(0, 4)


