import os
import json
import codecs
import config
from collections import namedtuple

from config.Config_worker import Config

DataObject = namedtuple('DataObject', ['description', 'symptoms'])


class DataLoader:
    def __init__(self, *, config_data, data_path):
        self.config_data = config_data
        self.data_path = data_path

    def load_chunks(self):
        data = []
        empty_objects = 0
        for files in os.listdir(self.data_path):
            if files.endswith('json'):
                file = os.path.join(self.data_path, files)
                med_notes = json.load(codecs.open(file, 'r', 'utf-8-sig'))
                for person in med_notes:
                    for desc in med_notes[person]:

                        if desc[1]:
                            data.append(DataObject(desc[0], desc[1]))
                        else:
                            empty_objects += 1
        print('Empty_sympt:', empty_objects)
        return data


if __name__ == '__main__':
    experiment_config = Config(file_name=config.CONFIG_PATH)
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../') + '/data/raw_data/')
    data_loader = DataLoader(config_data=experiment_config, data_path=data_path)
    data_loader.load_chunks()
