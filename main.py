from data.loader import FileIO
from util.conf import ModelConf
import time

class Rec(object):
    def __init__(self, config):
        self.config = config
        self.training_data = FileIO.load_data_set(config['training.set'], config['model']['type'])
        self.test_data = FileIO.load_data_set(config['test.set'], config['model']['type'])

        self.kwargs = {}
        print('Reading data and preprocessing...')

    def execute(self):
        import_str = f"from model.{self.config['model']['type']}.{self.config['model']['name']} import {self.config['model']['name']}"
        exec(import_str)
        recommender = f"{self.config['model']['name']}(self.config,self.training_data,self.test_data,**self.kwargs)"
        eval(recommender).execute()

if __name__ == '__main__':
    model = 'GCLCP'
    s = time.time()
    conf = ModelConf(f'./conf/{model}.yaml')
    rec = Rec(conf)
    rec.execute()
    e = time.time()
    print(f"Running time: {e - s:.2f} s")
