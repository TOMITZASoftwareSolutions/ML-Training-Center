from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Model
import hashlib
import json
from utils.model_factory import save_model_and_generate_id

model = VGG16(weights=None)
model.name = 'Distance Estimator VGGNet16'
adam = Adam()
model.compile(adam, loss='categorical_crossentropy')

save_model_and_generate_id(model)
