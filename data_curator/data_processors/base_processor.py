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
    def __init__(self, data, metadata, data_checks, params):
        self.data = data
        self.metadata = metadata
        self.data_checks = data_checks
        self.params = params
        self.processed_data = None
        self.initiate_processed_output()

        self.removed_cols = []
        self.main_data_key = self.metadata['main_data_key']
        self.second_data_key = self.metadata['second_data_key']
        self.main_target_col = self.metadata['main_target_col']
        self.column_checks = self.data_checks['column_checks']

    def initiate_processed_output(self):
        for key, dataset in self.data.items():
            self.processed_data[key] = self.data[key].copy(deep=True)

    def remove_cardinality_issue_columns(self):

        cardinality_checks = self.column_checks[self.main_data_key]['CARDINALITY_CHECK']
        cardinality_issue_cols = [key for key, value in cardinality_checks.items()
                                  if ((value == 'FAIL') and (key != self.main_target_col))]
        self.processed_data[self.main_data_key] = self.processed_data[self.main_data_key].drop(
            columns=cardinality_issue_cols)
        logger.info('removed cardinality issue columns {0}'.format(cardinality_issue_cols))
        if self.second_data_key != 'null':
            self.processed_data[self.second_data_key] = self.processed_data[self.second_data_key].drop(
                columns=cardinality_issue_cols
            )
            logger.info('removed cardinality issue columns {0} from secondary dataset ({1} data)'.format(
                cardinality_issue_cols, self.second_data_key))

    def remove_duplicated_columns(self):

        duplication_checks = self.column_checks[self.main_data_key]['DUPLICATE_CHECK']
        duplication_issue_cols = [key for key, value in duplication_checks.items()
                                  if ((value == 'FAIL') and (key != self.main_target_col))]
        self.processed_data[self.main_data_key] = self.processed_data[self.main_data_key].drop(
            columns=duplication_issue_cols
        )
        logger.info('removed cardinality issue columns {0} from primary data ({1} data)'.format(
            duplication_issue_cols, self.main_data_key))
        if self.second_data_key != 'null':
            self.processed_data[self.second_data_key] = self.processed_data[self.second_data_key].drop(
                columns=duplication_issue_cols
            )
            logger.info('removed cardinality issue columns {0} from secondary dataset ({1} data)'.format(
                duplication_issue_cols, self.second_data_key))

    def process_missing_values(self):
        if self.metadata['split_type'] == 'train_test':
            self.process_missing_values_train_test()
        elif self.metadata['split_type'] == 'total':
            logger.info('imputation occurs before train-test split. beware of a possible data leak')
            self.process_missing_values_total()
        else:
            raise NotImplementedError

    def impute_missing_numeric(self, columns, fit='train', transform=None):
        if columns.__len__() == 0:
            return 1
        if transform is None:
            transform = ['train', 'test']

        imputer = SimpleImputer(missing_values=np.nan, strategy=self.params['numeric_imputation_method'])
        imputer.fit(self.processed_data[fit][columns].values)
        logger.debug('imputation transformer created using {0} data'.format(fit))
        for trans in transform:
            self.processed_data[trans][columns] = imputer.transform(self.processed_data[trans][columns].values)
            logger.debug('{0} data transformed using imputer'.format(trans))

    def impute_missing_categorical(self, columns, fit='train', transform=None):
        if columns.__len__() == 0:
            return 1
        if transform is None:
            transform = ['train', 'test']

        imputer = SimpleImputer(missing_values=np.nan, strategy=self.params['categorical_imputation_method'])
        imputer.fit(self.processed_data[fit][columns].values)
        logger.debug('imputation transformer created using {0} data'.format(fit))
        for trans in transform:
            self.processed_data[trans][columns] = imputer.transform(self.processed_data[trans][columns].values)
            logger.debug('{0} data transformed using imputer'.format(trans))

    def process_missing_values_train_test(self):
        missing_checks = self.column_checks[self.main_data_key]['MISSING_VALUE_CHECK']
        missing_columns = [key for key, value in missing_checks.items()
                           if ((value == 'FAIL') and (key != self.main_target_col))]
        missing_columns = [t for t in missing_columns if t not in self.removed_cols]
        missing_columns_numeric = [t for t in missing_columns if
                                   self.metadata['feature_dtypes'][self.main_data_key][t] in ['float', 'int']]
        self.impute_missing_numeric(columns=missing_columns_numeric, fit='train', transform=['train', 'test'])

        missing_columns_categorical = [t for t in missing_columns if
                                       self.metadata['feature_dtypes'][self.main_data_key][t] in ['string']]
        self.impute_missing_categorical(columns=missing_columns_categorical, fit='train', transform=['train', 'test'])

    def process_missing_values_total(self):
        missing_checks = self.column_checks[self.main_data_key]['MISSING_VALUE_CHECK']
        missing_columns = [key for key, value in missing_checks.items()
                           if ((value == 'FAIL') and (key != self.main_target_col))]
        missing_columns = [t for t in missing_columns if t not in self.removed_cols]
        missing_columns_numeric = [t for t in missing_columns if
                                   self.metadata['feature_dtypes'][self.main_data_key][t] in ['float', 'int']]
        self.impute_missing_numeric(columns=missing_columns_numeric, fit='total', transform=['total'])

        missing_columns_categorical = [t for t in missing_columns if
                                       self.metadata['feature_dtypes'][self.main_data_key][t] in ['string']]
        self.impute_missing_categorical(columns=missing_columns_categorical, fit='total', transform=['total'])

    def encode_categorical_features(self):
        pass

    def process_memory_issues(self):
        pass

    def process_train_test_dtypes_issues(self):
        pass


class RegressionDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks, params):
        super().__init__(data, metadata, data_checks, params)


class ClassificationDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks, params):
        super().__init__(data, metadata, data_checks, params)


class UnsupervisedDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks, params):
        super().__init__(data, metadata, data_checks, params)
