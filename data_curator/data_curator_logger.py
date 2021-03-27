import logging

class DataCuratorLogger(object):
    def __init__(self, level):
        self.level = level
        self.get_log_level()
        self.set_log_level()
        
    def get_log_level(self):
        if self.level == 'DEBUG':
            self.loglevel = logging.DEBUG
        elif self.level == 'INFO':
            self.loglevel = logging.INFO
        else:
            raise ValueError('logging level not understood. valid values: DEBUG, INFO')
    
    def set_log_level(self):
        logging.basicConfig(level = self.loglevel)
