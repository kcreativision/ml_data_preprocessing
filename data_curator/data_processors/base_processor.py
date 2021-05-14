import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


def dtype_mapper(dtype):
    base_mapping = {
        np.float16:    'numeric',
        np.float32:    'numeric',
        np.float64:    'numeric',
        np.float128:   'numeric',
        np.int8:       'numeric',
        np.int16:      'numeric',
        np.int32:      'numeric',
        np.int64:      'numeric',
    }


class BaseDataProcessor(object):
    def __init__(self, data, metadata, data_checks):
        self.data = data
        self.metadata = metadata
        self.data_checks = data_checks
        self.processed_data = None
        self.initiate_processed_output()

    def initiate_processed_output(self):
        for key, dataset in self.data.items():
            self.processed_data[key] = self.data[key].copy(deep=True)

    def process_missing_values_issues(self):
        if self.metadata['split_type'] == 'train_test':
            self.process_missing_values_issues_train_test()
        elif self.metadata['split_type'] == 'total':
            logger.info('imputation occurs before train-test split. beware of a possible data leak')
            self.process_missing_values_issues_total()
        else:
            raise NotImplementedError

    def process_missing_values_issues_train_test(self):
        for key in self.data.keys():
            for col in self.data[key].columns:
                logger.debug(key + 'missing value imputation for {0}'.format(col))
                if self.data_checks['column_checks'][key]['MISSING_VALUE_CHECK'][col] == 'PASS':
                    pass
                else:
                    if self.metadata['feature_dtypes'][key][col]:
                        # TODO
                        pass

    def process_missing_values_issues_total(self):
        pass

    def process_cardinality_issues(self):
        pass

    def process_memory_issues(self):
        pass

    def process_train_test_dtypes_issues(self):
        pass


class RegressionDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks):
        super().__init__(data, metadata, data_checks)


class ClassificationDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks):
        super().__init__(data, metadata, data_checks)


class UnsupervisedDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks):
        super().__init__(data, metadata, data_checks)
