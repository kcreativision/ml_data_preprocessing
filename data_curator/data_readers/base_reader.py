import os
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

valid_exts = ['.csv']


def nullify_empty(df):
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df


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
        self.data = dict()
        for key, filename in self.data_info['datafiles'].items():
            self.data[key] = self.read_data_by_ext(filename, self.data_info['ext'][key])
    
    def read_data_by_ext(self, filename, extension):
        # TODO call CsvReader/ParquetReader/APIReader, etc. based on filename
        # TODO make sure to return a pandas dataframe only as subsequent ops depend on it
        if extension == '.csv':
            logger.debug(filename + ' reading completed')
            file_read = pd.read_csv(filename)
            file_read = nullify_empty(file_read)
            return file_read
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
            self.data_info['features'][key] = self.data[key].columns
            self.data_info['target_col'][key] = None


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
        return self.data, self.data_info

    def compare_train_test_ext(self):
        assert self.data_info['ext']['train'] == self.data_info['ext']['test']
        logger.debug('PASSED: train test file extensions same')

    def compare_train_test_cols(self):
        assert set(self.data_info['features']['train']) == set(self.data_info['features']['test']), \
            'different columns in train and test data'
        if self.data_info['target_col']['test'] is not None:
            assert self.data_info['target_col']['train'] == self.data_info['target_col']['test']
        logger.debug('PASSED: train test features and target col same/checked')

    def establish_cols_with_target(self):
        logger.debug('checking for target column: {0}'.format(self.data_info['target_col']))
        self.data_info['features'] = dict()
        self.data_info['target_col'] = dict()

        target_col = self.data_info['target_col']
        self.data_info['target_col']['train'] = target_col

        assert target_col in self.data['train'].columns, \
            'target column: {0} not present in training data'.format(target_col)
        self.data_info['features']['train'] = self.data['train'].columns.tolist().remove(target_col)

        if target_col not in self.data['test'].columns:
            logger.info('No target column in test data. Seems like competition data.')
            self.data_info['target_col']['test'] = None
            self.data_info['features']['test'] = self.data['test'].columns.tolist()
        else:
            logger.debug('target column present in test data.')
            self.data_info['target_col']['test'] = target_col
            self.data_info['features']['test'] = self.data['test'].columns.tolist().remove(target_col)
        

class TotalDataReader(BaseDataReader):
    def __init__(self, data_info):
        super().__init__(data_info)
    
    def run(self):
        self.get_ext()
        self.validate_ext()
        self.read_data()
        self.establish_cols()
        return self.data, self.data_info

    def establish_cols_with_target(self):
        self.data_info['features'] = dict()
        self.data_info['target_col'] = dict()

        target_col = self.data_info['decision_variable']
        self.data_info['target_col']['total'] = target_col

        assert target_col in self.data['total'].columns, \
            'target column: {0} not present in total data'.format(target_col)
        self.data_info['features']['total'] = self.data['total'].columns.tolist().remove(target_col)
        logger.debug('fetched target column: {0}'.format(target_col))
