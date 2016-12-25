from model.nnmodel import NNModel
from datetime import datetime
import rx
from rx import Observable, Observer
import concurrent.futures


class TrainingScheduler:
    def __init__(self, model_repository, training_manager):
        self.model_repository = model_repository
        self.training_manager = training_manager

    def add(self, model_name):
        model = self.model_repository.get_by_name(model_name)

        if model:
            if not model.active:
                model.active = True
                model.datamodified = datetime.now()
        else:
            model = NNModel.create_inst(model_name, datecreated=datetime.now(), datemodified=datetime.now(), epoch=0,
                                        active=True)
        self.model_repository.save(model)
        self.run()

    def remove(self, model_name):
        model = self.model_repository.get_by_name(model_name)
        if model:
            model.active = False
            model.datemodified = datetime.now()
            self.model_repository.save(model)

            running_model = self.training_manager.get_current()
            if running_model and running_model.name == model_name:
                self.training_manager.stop_training()

        self.run()

    def train(self, epochs=10):
        pass

    def get_current(self):
        current_model = self.training_manager.get_current()
        return current_model

    def run(self):
        if not self.training_manager.get_current():
            model = self.model_repository.get_next()

            if model:
                self.training_manager.train(model, epochs=1).subscribe(self.train_update, self.train_error,
                                                                       self.train_completed)

    def train_update(self, model):
        print 'Train start for model: {0};Current epoch: {1}'.format(model.name, model.epoch)
        model.epoch += 1
        model.datemodified = datetime.now()
        self.model_repository.save(model)
        print 'Train update for model: {0}; Finished epochs:{1}'.format(model.name, model.epoch)

    def train_error(self, e):
        print "Train error" + e.message
        self.run()

    def train_completed(self):
        print "Train completed"
        self.run()
