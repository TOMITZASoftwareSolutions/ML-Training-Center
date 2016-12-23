from peewee import *
from repository.database_helper import DatabaseHelper


class BaseModel(Model):
    class Meta:
        database = DatabaseHelper().database
