from unittest import TestCase

from managers.config_manager import ConfigManager
from managers.training_manager import TrainingManager
from model.nnmodel import NNModel
from datetime import datetime


class TestTraining(TestCase):
    def test_training(self):
        config_manager = ConfigManager(data_folder='../test_data', models_folder='../test_models')
        training_manager = TrainingManager(config_manager=config_manager)

        model = NNModel.create_inst('df6ad8d24088d13c090738019f0299428f466fb99385c42b1fa6cd32f7c70b79', datetime.now(), datetime.now(), epoch=0, active=True)

        training_manager.train_model(model)
