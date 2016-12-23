import repository.basemodel as basemodel
from model.nnmodel import NNModel

database_helper = basemodel.DatabaseHelper()
database_helper.create_tables([NNModel])
