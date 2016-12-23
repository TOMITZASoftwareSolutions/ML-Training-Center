from unittest import TestCase

from model.nnmodel import NNModel
from repository.basemodel import DatabaseHelper
from repository.model_repository import ModelRepository
import datetime


class test_repository(TestCase):
    def setUp(self):
        database_helper = DatabaseHelper()
        database_helper.create_tables([NNModel])

    def tearDown(self):
        pass

    def test_insert(self):
        model_repository = ModelRepository()

        model = model_repository.get_next()
        self.assertEqual(model, None)

        models = model_repository.getAll()
        self.assertEqual(len(models), 0)

        model = NNModel.create_inst(name='test', datemodified=datetime.datetime.now(),
                                    datecreated=datetime.datetime.now(), epoch=1, active=False)
        model_repository.save(model)

        models = model_repository.getAll()
        self.assertEqual(len(models), 1)

        model = model_repository.get_by_name('test')
        self.assertEqual(model.epoch, 1)

        model.epoch = 2
        model_repository.save(model)

        model = model_repository.get_by_name('test')
        self.assertEqual(model.epoch, 2)

        model2 = NNModel.create_inst(name='test2', datemodified=datetime.datetime.now(),
                                     datecreated=datetime.datetime.now(), epoch=0, active=True)
        model_repository.save(model2)

        model3 = NNModel.create_inst(name='test3', datemodified=datetime.datetime.now(),
                                     datecreated=datetime.datetime.now(), epoch=0, active=True)

        model_repository.save(model3)

        models = model_repository.getAll()
        self.assertEqual(len(models), 3)

        model.active = True
        model.datemodified = datetime.datetime.now()
        model_repository.save(model)

        model = model_repository.get_next()
        self.assertEqual(model.active, True)
        self.assertEqual(model.name, 'test2')
