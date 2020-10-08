import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import random
import glob

from keras.models import model_from_json

model = None

with open('model/model.json', 'r') as f:
  model = model_from_json(f.read())

model.load_weights('model/model_weights.h5')

print(model)


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT)
IMG_INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)


test_dataset = glob.glob('data/test/' + '*')


def get_image(sample_dir, shape=IMG_SHAPE, color=1):
  path = sample_dir + '/images/' + sample_dir.split('/')[-1] + '.png'
  image =  cv2.imread(path, color)
  return cv2.resize(image, shape) if shape else image


for image_path in test_dataset:

	item = image_path.split('/')[-1]

	image_orig = get_image(image_path, shape=None)
	image = get_image(image_path) / 255

	preds = (model.predict(np.array([image])) > 0.5).astype(np.uint8)
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

	pred = preds[0][..., 0]

	#dice = round(np.sum(mask[pred==1])*2.0 / (np.sum(mask) + np.sum(pred)),2)

	axes[0].imshow(image_orig)
	axes[0].set_title('Original')
	axes[1].imshow(image)
	axes[1].set_title('Original (resized)')
	axes[2].imshow(pred, cmap='gray')
	axes[2].set_title(f'Predicted')

	plt.savefig('predictions/' + item)