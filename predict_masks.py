import matplotlib.pyplot as plt
import numpy as np
import glob

from model import model
from functions import get_image, get_mask


test_dataset = glob.glob('data/test/' + '*')


for image_path in test_dataset:

  item = image_path.split('/')[-1]

  image_orig = get_image(image_path, shape=None)
  image = get_image(image_path) / 255

  preds = (model.predict(np.array([image])) > 0.5).astype(np.uint8)
  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

  pred = preds[0][..., 0]

  axes[0].imshow(image_orig)
  axes[0].set_title('Original')
  axes[1].imshow(image)
  axes[1].set_title('Original (resized)')
  axes[2].imshow(pred, cmap='gray')
  axes[2].set_title(f'Predicted')

  plt.savefig('predictions/' + item)
  plt.close()