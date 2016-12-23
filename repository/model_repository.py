from model.nnmodel import NNModel
from peewee import *


class ModelRepository():
    def get_by_name(self, name):
        try:
            model = NNModel().get(NNModel.name == name)
        except NNModel.DoesNotExist:
            model = None
        return model

    def getAll(self):
        return [model for model in NNModel.select()]

    def save(self, model):
        try:
            model.save(force_insert=True)
        except IntegrityError:
            model.save(force_insert=False)
        return model

    def get_next(self):
        try:
            model = NNModel().select().where(NNModel.active == True).order_by(NNModel.datemodified.asc()).get()
        except NNModel.DoesNotExist:
            model = None

        return model
