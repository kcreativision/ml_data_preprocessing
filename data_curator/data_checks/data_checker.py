import logging
logger = logging.getLogger(__name__)


class DataChecker(object):
    def __init__(self, data):
        self.data = data

    def check_missing_values(self):
        pass 
    
    def check_cardinality(self):
        pass
    
    def check_validation_split(self):
        pass
    
    def check_class_balance(self):
        pass
    
    def check_memory_issue(self):
        pass
    
    def check_frequency(self):
        pass
