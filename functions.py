import numpy as np
import cv2
import random
import glob

from config import * 


def get_image(sample_dir, shape=IMG_SHAPE, color=1):
  path = sample_dir + '/images/' + sample_dir.split('/')[-1] + '.png'
  image =  cv2.imread(path, color)
  return cv2.resize(image, shape) if shape else image


def get_mask(sample_dir, shape=IMG_SHAPE, color=0):
  all_masks = glob.glob(sample_dir + '/masks/*')
  image_mask = cv2.imread(all_masks[0], color)
  for mask in all_masks[1:]:
    image_mask = np.maximum(image_mask, cv2.imread(mask, color))
  return cv2.resize(image_mask, shape) if shape else image_mask


def data_train_generator(all_data_dirs, batch_size):
  while True:
    x_batch = []
    y_batch = []
    for i in range(batch_size):
      sample = random.choice(all_data_dirs)
      x_batch += [get_image(sample)]
      y_batch += [get_mask(sample)]
    x_batch = np.array(x_batch) / 255.
    y_batch = np.array(y_batch) / 255.
    yield x_batch, np.expand_dims(y_batch, -1)