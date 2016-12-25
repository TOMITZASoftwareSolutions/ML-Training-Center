from keras.models import Model, Sequential
from keras.layers import Dense, Flatten
from utils.model_factory import save_model_and_generate_id

model = Sequential()
model.add(Flatten(input_shape=(224, 224, 3)))
model.add(Dense(2))
model.add(Dense(2))

model.compile(optimizer='adam', loss='categorical_crossentropy')

save_model_and_generate_id(model, 32, save_location='tests/test_models')
