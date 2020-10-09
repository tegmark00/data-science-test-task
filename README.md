# Data Science task


Semantic​ segmentation model with UNet architecture using Keras.
Model builded on the dataset which contains a large number of segmented nuclei images


(more info: https://www.kaggle.com/c/data-science-bowl-2018/)


## Notebook

You may see model, data analys, etc in google colab notebook, located at `google-colab-notebooks` or just visit https://colab.research.google.com/drive/1k0V8bMGDUocoq4klF5YYF-TzSfJkGrr0?usp=sharing

## How to install

In order to run model and install dependencies use these commands:

1. Create virtual environment
 - $ `pyhon3 -m venv venv`

2. Activate your virtual environment
 - (venv) $ `source venv/bin/activate`

3. Install dependencies
 - (venv) $ `pip install -r requrements.txt`

Now you can run scripts:

4. Download and extract the dataset
 - (venv) $ python getdataset.py

After script execution you'll see two folders in data with `test` and `train` data

`train` data containes images and its mask

`test` only images

5. In order to predict masks for images in test folder, execute:
 - (venv) $ python predict_masks.py

6. If you want to train neural network, run:
 - (venv) $ python train.py

Notice that model already trained. Model and weights located in `model` folder