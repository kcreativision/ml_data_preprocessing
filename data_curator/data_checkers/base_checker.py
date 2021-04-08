import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
pd.options.mode.use_inf_as_na = True


class BaseDataChecker(object):
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.data_checks = dict()
        self.initiate_check_output()

    def initiate_check_output(self):
        self.data_checks['column_checks'] = dict()
        for key, dataset in self.data.items():
            self.data_checks['column_checks'][key] = pd.DataFrame(columns=dataset.columns)
        # TODO add another dataframe for trait vs test relative comparison output

    def check_missing_values(self):
        """
        returns column-wise PASS/FAIL:
        FAIL: at least one value is np.NaN or empty string or np.inf and 
            not all values are missing (as this is checked in cardinality check)
        PASS: Otherwise
        """
        for key in self.data.keys():
            any_missing_val_check = self.data[key].isnull().any(axis=0).values
            notall_missing_val_check = np.invert(self.data[key].isnull().all(axis=0).values)
            final_check = np.multiply(any_missing_val_check, notall_missing_val_check)
            missing_val_check = ['FAIL' if t else 'PASS' for t in final_check]
            self.data_checks['column_checks'][key].loc['MISSING_VALUE_CHECK'] = missing_val_check

    def check_cardinality(self):
        """
        returns column-wise PASS/FAIL:
            FAIL: if completely unique or single unique value or empty
            PASS: Otherwise
        """
        for key in self.data.keys():
            max_cardinal_cols = np.equal(self.data[key].nunique(axis=0).values,
                                         self.data[key].shape[0])
            min_cardinal_cols = np.equal(self.data[key].nunique(axis=0).values, 1)
            complete_missing_cols = self.data[key].isnull().all(axis=0).values

            cardinality_boolean = np.logical_or(max_cardinal_cols,
                                                np.logical_or(min_cardinal_cols,
                                                              complete_missing_cols))
            cardinality_check = ['FAIL' if t else 'PASS' for t in cardinality_boolean]

            self.data_checks['column_checks'][key].loc['CARDINALITY_CHECK'] = cardinality_check

    def check_validation_split(self):
        '''
        This outputs recommendation of validation_type(cross-validation, split) and validation parameter
        for either total or train
        '''
        self.data_checks['validation_reco'] = dict()
        num_samples = self.data[self.metadata['main_data_key']].shape[0]
        if num_samples > 20000:
            self.data_checks['validation_reco']['val_type'] = 'train_test_split'
            # 10% of data held for validation and metric tuning
            self.data_checks['validation_reco']['val_size'] = 0.1
        elif num_samples <= 20000:
            self.data_checks['validation_reco']['val_type'] = 'cross_val'
            self.data_checks['validation_reco']['val_size'] = 10 if num_samples < 1000 else 3

    def train_test_dtypes(self):
        """
        testing only for train_test type:
        returns column-wise PASS/FAIL:
            PASS: if column dtype is same between train and test datasets
            FAIL: Otherwise
        """
        if self.metadata['split_type'] == 'total':
            return 1
        else:
            for test_feature in self.metadata['features']['test']:
                check_bool = self.metadata['feature_dtypes']['train'][test_feature] == \
                             self.metadata['feature_dtypes']['test'][test_feature]
                self.data_checks['column_checks']['test'].loc['DTYPE_CHECK', test_feature] = \
                    'PASS' if check_bool else 'FAIL'

    def check_memory_issue(self):
        """figure out if the total computation exceeds the memory limit"""
        pass


class RegressionDataChecker(BaseDataChecker):
    def __init__(self, data, metadata):
        super().__init__(data, metadata)

    def run(self):
        self.check_missing_values()
        self.check_cardinality()
        self.check_validation_split()
        self.check_memory_issue()
        self.check_frequency()
        self.train_test_dtypes()
        return self.metadata, self.data_checks

    def check_frequency(self):
        """time series forecasting problems with datetime column can use this feature"""
        pass


class ClassificationDataChecker(BaseDataChecker):
    def __init__(self, data, metadata):
        super().__init__(data, metadata)

    def run(self):
        self.check_missing_values()
        self.check_cardinality()
        self.check_validation_split()
        self.check_class_balance()
        self.check_memory_issue()
        self.train_test_dtypes()
        return self.metadata, self.data_checks

    def check_class_balance(self):
        # TODO fill this - critical
        pass


class UnsupervisedDataChecker(BaseDataChecker):
    def __init__(self, data, metadata):
        super().__init__(data, metadata)

    def run(self):
        # TODO add check for only 'total' type as 'train_test' might not make sense here
        self.check_missing_values()
        self.check_cardinality()
        self.check_validation_split()
        self.check_memory_issue()
        return self.metadata, self.data_checks
