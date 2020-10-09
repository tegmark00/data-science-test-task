import keras
from keras import backend as K
import glob


from model import model
from functions import data_train_generator


from config import *


def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


last_w = keras.callbacks.ModelCheckpoint('model/model_weights.h5',
                                          monitor='val_loss',
                                          verbose=0,
                                          save_best_only=False,
                                          save_weights_only=True,
                                          mode='auto',
                                          save_freq=1)


callbacks = [last_w]


model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = [dice_coef])


train_dataset = glob.glob('data/train/*')


model.fit(
  data_train_generator(train_dataset, BATCH_SIZE),
  steps_per_epoch=STEPS_PER_EPOCH,
  epochs=EPOCHS,
  verbose=1,
  callbacks=callbacks,
  validation_data=data_train_generator(train_dataset, VALIDATION_BATCH_SIZE),
  validation_steps=VALIDATION_STEPS,
)