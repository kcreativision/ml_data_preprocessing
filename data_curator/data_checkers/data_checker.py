from data_curator.data_checkers.base_checker import UnsupervisedDataChecker, ClassificationDataChecker, \
    RegressionDataChecker
import data_curator.utils.data_checker_utils as dcu
import numpy as np
import logging
logger = logging.getLogger(__name__)


def get_data_keys(data_type, target_col_dict):
    if data_type == 'train_test':
        return 'train', target_col_dict['train'], 'test'
    elif data_type == 'total':
        return 'total', target_col_dict['total'], None
    else:
        raise NotImplementedError


class DataChecker(object):
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.checker_class = None
        self.metadata['main_data_key'], self.metadata['main_target_col'], self.metadata['second_data_key'] = \
            get_data_keys(self.metadata['split_type'], self.metadata['target_col'])
        self.metadata['cat_to_num_threshold'] = 100
        self.get_feature_dtypes()
    
    def get_feature_dtypes(self):
        self.metadata['feature_dtypes'] = dict()
        for data_key in [self.metadata['main_data_key'], self.metadata['second_data_key']]:
            if data_key is None:
                continue
            main_dtypes = [dcu.get_base_types(t) for t in self.data[data_key].dtypes.values]
            _cols_below_threshold_inds = list(np.array(self.data[data_key].nunique()) <=
                                              self.metadata['cat_to_num_threshold'])
            main_dtypes = [t+'_discreet' if _cols_below_threshold_inds[i] else t for i, t in enumerate(main_dtypes)]
            self.metadata['feature_dtypes'][data_key] = dict(zip(self.data[data_key].dtypes.index, 
                                                                 main_dtypes))

    def run(self):
        self.set_checker_type()
        return self.checker_class(self.data, self.metadata).run()

    def set_checker_type(self):
        if self.data_is_unsupervised():
            self.set_unsupervised_data_checker()
        if self.data_is_classification():
            self.set_classification_data_checker()
        else:
            # TODO check other cases
            self.set_regression_data_checker()
    
    def data_is_unsupervised(self):
        if self.metadata['main_target_col'] is None:
            return 1
        else:
            return 0

    def data_is_classification(self):
        train_data = self.data[self.metadata['main_data_key']]
        if (train_data[self.metadata['main_target_col']].dtype == 'object') or \
           (train_data[self.metadata['main_target_col']].unique().__len__() <= self.metadata['cat_to_num_threshold']):
            # TODO using arbitrary #classes, check for better solution
            return 1
        else:
            return 0
        
    def set_unsupervised_data_checker(self):
        self.metadata['learning_type'] = 'unsupervised'
        self.checker_class = UnsupervisedDataChecker
        logger.debug('data is of unsupervised learning format')
    
    def set_classification_data_checker(self):
        self.metadata['learning_type'] = 'classification'
        self.checker_class = ClassificationDataChecker
        logger.debug('data is of classification learning format')
    
    def set_regression_data_checker(self):
        self.metadata['learning_type'] = 'regression'
        self.checker_class = RegressionDataChecker
        logger.debug('data is of regression learning format')
