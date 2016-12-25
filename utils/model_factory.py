from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Model
import hashlib
import json


def hash_params(parameters):
    hashcode = hashlib.sha256(str(parameters).encode()).hexdigest()
    return hashcode


def get_model_optimizer_params(model):
    return {
        'optimizer_config': {
            'class_name': model.optimizer.__class__.__name__,
            'config': model.optimizer.get_config()
        },
        'loss': model.loss,
        'metrics': model.metrics,
        'sample_weight_mode': model.sample_weight_mode,
        'loss_weights': model.loss_weights,
    }


def save_model_and_generate_id(model, batch_size, save_location='saved_models'):
    model_architecture = model.get_config()
    model_optimizer = get_model_optimizer_params(model)

    model_configuration = dict()
    model_configuration['architecture'] = model_architecture
    model_configuration['optimizer'] = model_optimizer
    model_configuration['batch_size'] = batch_size

    model_id = hash_params(model_configuration)

    print model_id

    model_path = '{0}/{1}'.format(save_location, model_id)

    model.name = model_id
    model.save(model_path)
    with open('{0}.config'.format(model_path), 'w') as f:
        json.dump(model_configuration, f)
