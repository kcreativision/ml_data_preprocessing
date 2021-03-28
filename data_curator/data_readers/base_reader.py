import os
import pandas as pd

import logging
logger = logging.getLogger(__name__)


valid_exts = ['.csv']
class BaseDataReader(object):
    def __init__(self, data_info):
        self.data_info = data_info

    def get_ext(self):
        self.data_info['ext'] = dict()
        for key, filename in self.data_info['datafiles'].items():
            self.data_info['ext'][key] = os.path.splitext(filename)[1]

    def validate_ext(self):
        for key in self.data_info['datafiles'].keys():
            assert self.data_info['ext'][key] in valid_exts

    def read_data(self):
        self.data_info['data'] = dict()
        for key, filename in self.data_info['datafiles'].items():
            self.data_info['data'][key] = self.read_data_by_ext(filename, self.data_info['ext'][key])
    
    def read_data_by_ext(self, filename, extension):
        # TODO call CsvReader/ParquetReader/APIReader, etc. based on filename
        if extension == '.csv':
            logger.debug(filename + ' reading completed')
            return pd.read_csv(filename)
        else:
            raise NotImplementedError

    def establish_cols(self):
        if self.data_info['no_target']:
            self.establish_cols_no_target()
        else:
            self.establish_cols_with_target()

    def establish_cols_no_target(self):
        logger.debug('target column not present')
        self.data_info['features'] = dict()
        self.data_info['target_col'] = dict()

        for key, filename in self.data_info['datafiles'].items():
            self.data_info['features'][key] = self.data_info['data'][key].columns
            self.data_info['target_col'][key] = None
    
    def establish_cols_with_target(self):
        logger.debug('last column in train data is the target column')
        self.data_info['features'] = dict()
        self.data_info['target_col'] = dict()
        
        self.data_info['features']['train'] = self.data_info['data']['train'].columns.tolist()[:-1]
        target_col = self.data_info['data']['train'].columns[-1]
        self.data_info['target_col']['train'] = target_col
        
        if target_col not in self.data_info['data']['test'].columns:
            logger.info('No target column in test data. Seems like competition data.')
            self.data_info['target_col']['test'] = None
            self.data_info['features']['test'] = self.data_info['data']['test'].columns.tolist()


class TrainTestDataReader(BaseDataReader):
    def __init__(self, data_info):
        super().__init__(data_info)
    
    def run(self):
        self.get_ext()
        self.validate_ext()
        self.compare_train_test_ext()
        self.read_data()
        self.establish_cols()
        self.compare_train_test_cols()
        logger.debug(self.data_info)

    def compare_train_test_ext(self):
        assert self.data_info['ext']['train'] == self.data_info['ext']['test']
        logger.debug('PASSED: train test file extensions same')

    def compare_train_test_cols(self):
        assert set(self.data_info['features']['train']) == set(self.data_info['features']['test']), \
              'different columns in train and test data'
        if self.data_info['target_col']['test'] is not None:
            assert self.data_info['target_col']['train'] == self.data_info['target_col']['test']
        logger.debug('PASSED: train test features and target col same/checked')
        

class TotalDataReader(BaseDataReader):
    def __init__(self, data_info):
        super().__init__(data_info)
    
    def run(self):
        self.get_ext()
        self.validate_ext()
        self.read_data()
        self.establish_cols()
        logger.debug(self.data_info)