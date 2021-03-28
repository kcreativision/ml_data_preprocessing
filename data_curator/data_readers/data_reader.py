from data_curator.data_readers.base_reader import TrainTestDataReader, TotalDataReader

import logging
logger = logging.getLogger(__name__)


class DataReader(object):
    def __init__(self, datafiles, no_target):
        self.data_info = dict()
        self.data_info['datafiles'] = datafiles
        self.data_info['no_target'] = True if no_target else False
    
    def run(self):
        self.set_reader_type()
        return self.reader_class(self.data_info).run()

    def set_reader_type(self):
        if ('train' in self.data_info['datafiles'].keys()) and \
            ('test' in self.data_info['datafiles'].keys()):
            self.set_train_test_reader()
        elif 'total' in self.data_info['datafiles'].keys():
            self.set_total_reader()
        else:
            raise NotImplementedError

    def set_train_test_reader(self):
          self.data_info['type'] = 'train_test'
          self.reader_class = TrainTestDataReader
          logger.debug('data is of two-files (train test) format')
    
    def set_total_reader(self):
          self.data_info['type'] = 'total'
          self.reader_class = TotalDataReader
          logger.debug('data is of single file(total) format')


    