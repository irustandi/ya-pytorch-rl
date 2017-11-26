from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import random
import torchvision.transforms as T
from PIL import Image
import numpy as np
import tensorboard_logger


# from http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ToGrayscale(object):

    def __init__(self):
        return

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Grayscale image.
        """
        return img.convert('L')


class Crop(object):

    def __init__(self, i, j, h, w):
        self.i = i
        self.j = j
        self.h = h
        self.w = w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Grayscale image.
        """
        return img.crop((self.j, self.i, self.j + self.w, self.i + self.h))


class ToNDArray(object):
    def __call__(self, img):
        data = np.asarray(img, dtype=np.uint8)
        data = data.T
        data = np.expand_dims(data, axis=2)
        return data


def get_transforms():
    return T.Compose([T.ToPILImage(),
                      ToGrayscale(),
                      Crop(34, 0, 160, 160),
                      T.Scale(84, interpolation=Image.NEAREST),
                      ToNDArray(),
                      T.ToTensor()])