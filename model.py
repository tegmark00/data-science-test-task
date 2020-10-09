from keras.models import model_from_json


model = None

with open('model/model.json', 'r') as f:
  model = model_from_json(f.read())

model.load_weights('model/model_weights.h5')