from data_curator.data_processors.base_processor import UnsupervisedDataProcessor, ClassificationDataProcessor, \
    RegressionDataProcessor
import logging
logger = logging.getLogger(__name__)


class DataProcessor(object):
    def __init__(self, data, metadata, data_checks, params):
        self.data = data
        self.metadata = metadata
        self.data_checks = data_checks
        self.params = params
        self.processor_class = None

    def run(self):
        self.set_checker_type()
        return self.processor_class(self.data, self.metadata, self.data_checks, self.params).run()

    def set_checker_type(self):
        if self.metadata['learning_type'] == 'unsupervised':
            self.set_unsupervised_data_checker()
        elif self.metadata['learning_type'] == 'classification':
            self.set_classification_data_checker()
        elif self.metadata['learning_type'] == 'regression':
            self.set_regression_data_checker()
        else:
            raise NotImplementedError

    def set_unsupervised_data_checker(self):
        self.processor_class = UnsupervisedDataProcessor
        logger.debug('data is of unsupervised learning format')

    def set_classification_data_checker(self):
        self.processor_class = ClassificationDataProcessor
        logger.debug('data is of classification learning format')

    def set_regression_data_checker(self):
        self.processor_class = RegressionDataProcessor
        logger.debug('data is of regression learning format')
