from peewee import *
from repository.basemodel import BaseModel


class NNModel(BaseModel):
    name = CharField(primary_key=True, max_length=256)
    datemodified = DateTimeField(null=False)
    datecreated = DateTimeField(null=False)
    epoch = IntegerField()
    active = BooleanField()


    @staticmethod
    def create_inst(name, datecreated, datemodified, epoch, active):
        return NNModel(name=name, datecreated=datecreated, datemodified=datemodified, epoch=epoch, active=active)
