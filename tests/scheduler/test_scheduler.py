from unittest import TestCase
from mockito import *
import time

from managers.training_manager import TrainingManager
from managers.training_scheduler import TrainingScheduler
from model.nnmodel import NNModel
from repository.database_helper import DatabaseHelper
from repository.model_repository import ModelRepository


class TestScheduler(TestCase):
    def setUp(self):
        database_helper = DatabaseHelper()
        database_helper.create_tables([NNModel])

    def test_scheduler(self):
        model_repository = ModelRepository()
        training_manager = TrainingManager()
        training_scheduler = TrainingScheduler(model_repository, training_manager)

        training_scheduler.add('test1')
        training_scheduler.add('test2')

        training_scheduler.run()
        current_running_model = training_scheduler.get_current()
        self.assertEqual(current_running_model.name, 'test1')

        time.sleep(2.2)

        current_running_model = training_scheduler.get_current()
        self.assertEqual(current_running_model.name, 'test2')

        time.sleep(2.2)

        current_running_model = training_scheduler.get_current()
        self.assertEqual(current_running_model.name, 'test1')

        training_scheduler.remove('test2')

        time.sleep(2.2)

        current_running_model = training_scheduler.get_current()
        self.assertEqual(current_running_model.name, 'test1')

        time.sleep(2.2)

        current_running_model = training_scheduler.get_current()
        self.assertEqual(current_running_model.name, 'test1')
        self.assertEqual(current_running_model.epoch, 3)

        model = model_repository.get_by_name('test2')
        self.assertEqual(model.active, False)

        time.sleep(4)
