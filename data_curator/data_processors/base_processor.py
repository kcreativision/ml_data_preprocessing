from sklearn.impute import SimpleImputer
import data_curator.utils.generic_utils as gu
import numpy as np
import pandas as pd
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
        self.processed_data = data
        self.metadata = metadata
        self.data_checks = data_checks
        self.params = params

        self.processed_info = dict()
        self.initiate_processed_output()

        self.removed_cols = []
        self.main_data_key = self.metadata['main_data_key']
        self.second_data_key = self.metadata['second_data_key']
        self.main_target_col = self.metadata['main_target_col']
        self.column_checks = self.data_checks['column_checks']

    def initiate_processed_output(self):
        self.processed_info['column_processing'] = dict()
        for key, dataset in self.processed_data.items():
            self.processed_info['column_processing'][key] = dict(zip(list(dataset.columns),
                                                                     ['PASS']*len(dataset.columns)))

        self.processed_info['row_processing'] = 'PASS'

    def _get_issue_cols(self, check_name, data_key='main_data_key'):
        if data_key == 'main_data_key':
            _checks = self.column_checks[self.main_data_key][check_name]
            _issue_cols = [key for key, value in _checks.items()
                                      if ((value == 'FAIL') and (key != self.main_target_col))]
            _issue_cols = [t for t in _issue_cols if t not in self.removed_cols]
        elif data_key == 'second_data_key':
            _checks = self.column_checks[self.second_data_key][check_name]
            _issue_cols = [key for key, value in _checks.items()
                           if ((value == 'FAIL') and (key != self.main_target_col))]
            _issue_cols = [t for t in _issue_cols if t not in self.removed_cols]
        else:
            raise NotImplementedError

        return _issue_cols

    def _update_processed_info_main_data(self, cols, message):
        _processed_info_message = [message]*len(cols)
        _processed_info_update = dict(zip(cols, _processed_info_message))
        self.processed_info['column_processing'][self.main_data_key].update(_processed_info_update)

    def _update_processed_info_second_data(self, cols, message):
        _processed_info_message = [message]*len(cols)
        _processed_info_update = dict(zip(cols, _processed_info_message))
        self.processed_info['column_processing'][self.second_data_key].update(_processed_info_update)

    def remove_cardinality_issue_columns(self):
        cardinality_issue_cols = self._get_issue_cols('CRITICAL_CARDINALITY_CHECK')
        self.processed_data[self.main_data_key] = self.processed_data[self.main_data_key].drop(
            columns=cardinality_issue_cols)
        self._update_processed_info_main_data(cardinality_issue_cols, 'REMOVED DUE TO CRITICAL CARDINALITY ISSUE')
        self.removed_cols.extend(cardinality_issue_cols)
        logger.info('removed critical cardinality issue columns {0}'.format(cardinality_issue_cols))
        if self.second_data_key is not None:
            self.processed_data[self.second_data_key] = self.processed_data[self.second_data_key].drop(
                columns=cardinality_issue_cols)
            self._update_processed_info_second_data(cardinality_issue_cols, 'REMOVED DUE TO CRITICAL CARDINALITY ISSUE')
            logger.info('removed critical cardinality issue columns {0} from secondary dataset ({1} data)'.format(
                cardinality_issue_cols, self.second_data_key))

    def remove_duplicated_columns(self):
        duplication_issue_cols = self._get_issue_cols('DUPLICATE_CHECK')
        self.processed_data[self.main_data_key] = self.processed_data[self.main_data_key].drop(
            columns=duplication_issue_cols
        )
        self.removed_cols.extend(duplication_issue_cols)
        self._update_processed_info_main_data(duplication_issue_cols, 'REMOVED DUE TO DUPLICATION ISSUE')
        logger.info('removed duplicated columns {0} from primary data ({1} data)'.format(
            duplication_issue_cols, self.main_data_key))
        if self.second_data_key is not None:
            self.processed_data[self.second_data_key] = self.processed_data[self.second_data_key].drop(
                columns=duplication_issue_cols
            )
            self._update_processed_info_second_data(duplication_issue_cols, 'REMOVED DUE TO DUPLICATION ISSUE')
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
        self.process_missing_target_var()

    def process_missing_values_train_test(self):
        missing_columns = self._get_issue_cols('MISSING_VALUE_CHECK')
        missing_columns_numeric = [t for t in missing_columns if
                                   self.metadata['feature_dtypes'][self.main_data_key][t] in ['float', 'int']]
        self.impute_missing_numeric(columns=missing_columns_numeric, fit='train', transform=['train', 'test'])
        missing_columns_categorical = [t for t in missing_columns if
                                       self.metadata['feature_dtypes'][self.main_data_key][t] in
                                       ['string_discreet', 'string', 'float_discreet', 'int_discreet']]
        self.impute_missing_categorical(columns=missing_columns_categorical, fit='train', transform=['train', 'test'])

        self._update_processed_info_main_data(missing_columns, 'MISSING VALUES IMPUTED')
        self._update_processed_info_second_data(missing_columns, 'MISSING VALUES IMPUTED')

    def process_missing_values_total(self):
        missing_columns = self._get_issue_cols('MISSING_VALUE_CHECK')
        missing_columns_numeric = [t for t in missing_columns if
                                   self.metadata['feature_dtypes'][self.main_data_key][t] in ['float', 'int']]
        self.impute_missing_numeric(columns=missing_columns_numeric, fit='total', transform=['total'])

        missing_columns_categorical = [t for t in missing_columns if
                                       self.metadata['feature_dtypes'][self.main_data_key][t] in
                                       ['string_discreet', 'string', 'float_discreet', 'int_discreet']]
        self.impute_missing_categorical(columns=missing_columns_categorical, fit='total', transform=['total'])

        self._update_processed_info_main_data(missing_columns, 'MISSING VALUES IMPUTED')

    def process_missing_target_var(self):
        if self.column_checks[self.main_data_key]['MISSING_VALUE_CHECK'][self.main_target_col] == 'PASS':
            return 1
        logger.debug('target col has missing values, removing corresponding rows')
        first_len = self.processed_data[self.main_data_key].shape[0]
        self.processed_data[self.main_data_key] = self.processed_data[self.main_data_key].\
            dropna(subset=[self.main_target_col], axis=0).\
            reset_index(drop=True)
        second_len = self.processed_data[self.main_data_key].shape[0]
        self.processed_info['row_processing'] = 'DROPPED NULL TARGET ROWS'
        logger.info('dropped null target rows. '
                    'resulted in {0:.2f}% drop in training size'.format(gu.drop_perc(first_len, second_len)))

    def impute_missing_numeric(self, columns, fit='train', transform=None):
        if columns.__len__() == 0:
            return 1
        if transform is None:
            transform = ['train', 'test']

        imputer = SimpleImputer(missing_values=np.nan, strategy=self.params['numeric_imputation_method'])
        imputer.fit(self.processed_data[fit][columns].values)
        logger.debug('numeric imputation transformer created using {0} data'.format(fit))
        for trans in transform:
            self.processed_data[trans][columns] = imputer.transform(self.processed_data[trans][columns].values)
            logger.debug('{0} data transformed using numeric imputer'.format(trans))

    def impute_missing_categorical(self, columns, fit='train', transform=None):
        if columns.__len__() == 0:
            return 1
        if transform is None:
            transform = ['train', 'test']

        imputer = SimpleImputer(missing_values=np.nan, strategy=self.params['categorical_imputation_method'])
        imputer.fit(self.processed_data[fit][columns].values)
        logger.debug('categorical imputation transformer created using {0} data'.format(fit))
        for trans in transform:
            self.processed_data[trans][columns] = imputer.transform(self.processed_data[trans][columns].values)
            logger.debug('{0} data transformed using categorical imputer'.format(trans))

    def encode_categorical_features(self):
        pass

    def process_memory_issues(self):
        pass

    def process_train_test_dtypes_issues(self):
        pass


class RegressionDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks, params):
        super().__init__(data, metadata, data_checks, params)

    def run(self):
        self.remove_cardinality_issue_columns()
        self.remove_duplicated_columns()
        self.process_missing_values()

        return self.processed_data, self.processed_info


class ClassificationDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks, params):
        super().__init__(data, metadata, data_checks, params)

    def run(self):
        self.remove_cardinality_issue_columns()
        self.remove_duplicated_columns()
        self.process_missing_values()

        return self.processed_data, self.processed_info


class UnsupervisedDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks, params):
        super().__init__(data, metadata, data_checks, params)

    def run(self):
        self.remove_cardinality_issue_columns()
        self.remove_duplicated_columns()
        self.process_missing_values()

        return self.processed_data, self.processed_info
