from peewee import SqliteDatabase


class DatabaseHelper:
    def __init__(self, database='models.db'):
        self.database = SqliteDatabase(database)

    def create_tables(self, models=[]):
        self.database.connect()
        self.database.drop_tables(models, safe=True)
        self.database.create_tables(models)
        self.database.close()

    def get_database(self):
        return self.database
