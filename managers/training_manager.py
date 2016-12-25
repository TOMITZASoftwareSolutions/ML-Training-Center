import json
from rx import Observable, Observer
from rx.concurrency import NewThreadScheduler, ThreadPoolScheduler
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from utils.tensorboard_callback import TensorBoard
import keras.backend as K


class TrainingManager(Observable):
    def __init__(self, config_manager):
        self.model = None
        self.keras_model = None
        self.config_manager = config_manager

    def train(self, model, epochs=1):
        self.model = model
        return self.subscribe_on(ThreadPoolScheduler(max_workers=1))

    def get_current(self):
        return self.model

    def subscribe(self, subscriber):
        try:
            self.train_model(self.model)
            if not self.keras_model.stop_training:
                subscriber.on_next(self.model)
            self.model = None
            self.keras_model = None
        except Exception as e:
            self.model = None
            self.keras_model = None
            subscriber.on_error(e)

        subscriber.on_completed()

    def stop_training(self):
        if self.keras_model:
            self.keras_model.stop_training = True

    def train_model(self, model):
        model_path = '{0}/{1}'.format(self.config_manager.models_folder, model.name)
        train_data_path = '{0}/train'.format(self.config_manager.data_folder)
        validation_data_path = '{0}/validate'.format(self.config_manager.data_folder)

        with open('{}.config'.format(model_path)) as fp:
            model_configuration = json.load(fp)

        self.keras_model = load_model(model_path)

        train_generator = ImageDataGenerator()
        validate_generator = ImageDataGenerator()

        print self.keras_model.input_shape

        image_target_size = self.keras_model.input_shape[1:3]

        train_generator = train_generator.flow_from_directory(train_data_path, target_size=image_target_size)
        validate_generator = validate_generator.flow_from_directory(validation_data_path, target_size=image_target_size)

        batch_size = model_configuration['batch_size']

        samples_per_epoch = train_generator.nb_sample
        nb_val_samples = validate_generator.nb_sample

        train_generator.batch_size = batch_size
        validate_generator.batch_size = batch_size

        tensor_board = TensorBoard(log_dir='logs/{}'.format(model.name), histogram_freq=5, write_images=True)

        self.keras_model.fit_generator(generator=train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=model.epoch+1,
                                       validation_data=validate_generator, nb_val_samples=nb_val_samples,
                                       callbacks=[tensor_board], max_q_size=1,initial_epoch=model.epoch)

        self.keras_model.save(model_path)

        K.clear_session()
