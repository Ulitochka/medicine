import os
import codecs as codecs
import json
from collections import OrderedDict


class Config:
    _config_data = None

    def __init__(self, file_name=None, config_string=None):
        if config_string is None:
            if file_name is None:
                file_path = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
                file_name = os.path.abspath(file_path + '/config.json')
            assert os.path.exists(file_name), "Configuration file " + file_name + " doesn't exist!"
            with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as f:
                config_data = json.load(f, object_pairs_hook=OrderedDict)
        else:
            config_data = json.loads(config_string, object_pairs_hook=OrderedDict)
        err_msg = "Configuration file is corrupt!"
        assert isinstance(config_data, OrderedDict), err_msg
        self._config_data = config_data

    def get(self, key):
        if key not in self._config_data:
            raise ValueError("Unable to load '%s' key from config" % key)
        return self._config_data[key]

    def set(self, key, value):
        self._config_data[key] = value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self._config_data)

    def __del__(self):
        del self._config_data
