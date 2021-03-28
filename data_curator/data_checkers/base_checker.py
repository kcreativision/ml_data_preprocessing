import pandas as pd
import logging
logger = logging.getLogger(__name__)


class BaseDataChecker(object):
    def __init__(self, data):
        self.data = data

    def run_checks(self):
        self.check_missing_values()
        self.check_cardinality()
        self.check_validation_split()
        self.check_class_balance()
        self.check_memory_issue()
        self.check_frequency()

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
    
    
