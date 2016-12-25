import argparse
from flask import Flask

from managers.config_manager import ConfigManager
from managers.training_manager import TrainingManager
from managers.training_scheduler import TrainingScheduler
from repository.model_repository import ModelRepository

app = Flask(__name__)

training_scheduler = None


@app.route('/')
def index():
    return 'The ML Training Center'


@app.route('/add/<model_id>', )
def add(model_id):
    training_scheduler.add(model_id)
    return 'Model added'


@app.route('/remove/<model_id>')
def remove(model_id):
    training_scheduler.remove(model_id)
    return 'Model removed'


@app.route('/train/<model_id>/<epochs>')
def train(model_id, epochs):
    return NotImplementedError('Please implement me...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default='tests/test_data')
    parser.add_argument("--models", default='tests/test_models')

    args = parser.parse_args()

    data_path = args.data
    models = args.models

    config_manager = ConfigManager(data_folder=data_path, models_folder=models)
    training_manager = TrainingManager(config_manager=config_manager)
    model_repository = ModelRepository()

    training_scheduler = TrainingScheduler(model_repository, training_manager)
    training_scheduler.run()

    app.local = True
    app.run(port=9191, host="0.0.0.0",
            debug=False)
