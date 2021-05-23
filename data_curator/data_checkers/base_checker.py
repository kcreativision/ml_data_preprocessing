import numpy as np
import pandas as pd
import logging
import itertools

logger = logging.getLogger(__name__)
pd.options.mode.use_inf_as_na = True


def get_range(col):
    if (col.dtype.str.startswith('<f')) or (col.dtype.str.startswith('<i')):
        return col.max() - col.min()
    else:
        return col.nunique()


class BaseDataChecker(object):
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.data_checks = dict()
        self.initiate_check_output()

    def initiate_check_output(self):
        self.data_checks['column_checks'] = dict()
        for key, dataset in self.data.items():
            self.data_checks['column_checks'][key] = dict()
        # TODO add another check for trait vs test relative comparison output

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
            self.data_checks['column_checks'][key]['MISSING_VALUE_CHECK'] = dict(zip(self.data[key].columns,
                                                                                     missing_val_check))

    def check_cardinality(self):
        self.check_critical_cardinality()
        self.check_low_cardinality()

    def check_critical_cardinality(self):
        """
        returns column-wise PASS/FAIL:
            FAIL: if completely unique or single unique value or empty
            PASS: Otherwise
        """
        for key in self.data.keys():
            max_cardinal_cols = np.equal(self.data[key].nunique(axis=0).values,
                                         self.data[key].shape[0])
            over_range_cols = self.data[key].apply(lambda x: get_range(x), axis=0) >= self.data[key].shape[0]
            min_cardinal_cols = np.equal(self.data[key].nunique(axis=0).values, 1)
            complete_missing_cols = self.data[key].isnull().all(axis=0).values

            cardinality_boolean = np.logical_or(np.logical_and(max_cardinal_cols, over_range_cols),
                                                np.logical_or(min_cardinal_cols, complete_missing_cols))
            cardinality_check = ['FAIL' if t else 'PASS' for t in cardinality_boolean]

            self.data_checks['column_checks'][key]['CRITICAL_CARDINALITY_CHECK'] = dict(zip(self.data[key].columns,
                                                                                            cardinality_check))

    def check_low_cardinality(self):
        """
        returns column-wise PASS/FAIL - ONLY for numeric features:
            FAIL: if low number of unique values specified by self.metadata['cat_to_num_threshold']
            PASS: Otherwise
        """
        for key in self.data.keys():
            numeric_cols = [key for key, val in self.metadata['feature_dtypes'][key].items()
                            if (val in ['float', 'int'])]
            other_cols = [t for t in self.data[key].columns if t not in numeric_cols]
            numeric_data = self.data[key][numeric_cols]
            low_cardinality_boolean = np.less_equal(numeric_data.nunique(axis=0).values,
                                                    self.metadata['cat_to_num_threshold'])
            low_cardinality_check = ['FAIL' if t else 'PASS' for t in low_cardinality_boolean]

            self.data_checks['column_checks'][key]['LOW_CARDINALITY_CHECK'] = dict(zip(numeric_cols,
                                                                                       low_cardinality_check))
            self.data_checks['column_checks'][key]['LOW_CARDINALITY_CHECK'].update(
                dict(zip(other_cols, ['Not_Applicable']*len(other_cols))))

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

    def check_train_test_dtypes(self):
        """
        testing only for train_test type:
        returns column-wise PASS/FAIL:
            PASS: if column dtype is same between train and test datasets
            FAIL: Otherwise
        """
        if self.metadata['split_type'] == 'total':
            return 1
        else:
            self.data_checks['column_checks']['test']['DTYPE_CHECK'] = dict()
            for test_feature in self.metadata['features']['test']:
                check_bool = self.metadata['feature_dtypes']['train'][test_feature] == \
                             self.metadata['feature_dtypes']['test'][test_feature]
                self.data_checks['column_checks']['test']['DTYPE_CHECK'][test_feature] = \
                    'PASS' if check_bool else 'FAIL'

    def check_duplicate_columns(self):

        def check_if_equal(ind0, ind1):
            col_names = [data.columns[ind0], data.columns[ind1]]
            df = data[col_names]
            df = df.dropna()
            return df[col_names[0]].equals(df[col_names[1]])

        def find_actual_base(bases):
            if len(bases) == 1:
                return bases[0]
            else:
                df = data[data.columns[bases]]
                not_na_proportions = df.notnull().sum(axis=0)
                col_ind_least_na = bases[not_na_proportions.argmax()]
                return col_ind_least_na

        for key in self.data.keys():
            data = self.data[key]
            col_combinations = [list(x) for x in itertools.combinations(np.arange(data.shape[1]), 2)]
            col_equals = [check_if_equal(col_ind1, col_ind2) for col_ind1, col_ind2 in col_combinations]
            col_duplicates = [col_combinations[i] for i, t in enumerate(col_equals) if t]
            duplicate_check = ['PASS'] * data.shape[1]
            self.data_checks['column_checks'][key]['DUPLICATE_CHECK'] = dict(zip(self.data[key].columns,
                                                                                 duplicate_check))
            for col1_ind, col2_ind in col_duplicates:
                # considering the leftmost column as the base
                col2_bases = [t0 for t0, t1 in col_duplicates if t1 == col2_ind]
                col2_base = find_actual_base(col2_bases)
                self.data_checks['column_checks'][key]['DUPLICATE_CHECK'][data.columns[col2_ind]] = 'FAIL'
                self.data_checks['column_checks'][key]['DUPLICATE_CHECK'][str(data.columns[col2_ind] + '_BASE')] = \
                    data.columns[col2_base]

    def check_target_var(self):
        """
        currently only highlights if any issue exists with target variable
        :return:
        """
        main_data_key = self.metadata['main_data_key']
        main_target_key = self.metadata['main_target_col']
        column_checks = self.data_checks['column_checks']
        for key, val in column_checks[main_data_key].items():
            _pass_status = val[main_target_key]
            if _pass_status in ['PASS', 'FAIL', 'Not_Applicable']:
                logger.info('{test} {status} for {col}'.format(test=key, status=_pass_status, col=main_target_key))
            else:
                raise ValueError('unknown status for target column {0} {1}'.format(key, val[main_target_key]))

    def check_memory_issue(self):
        """figure out if the total computation exceeds the memory limit"""
        pass


class RegressionDataChecker(BaseDataChecker):
    def __init__(self, data, metadata):
        super().__init__(data, metadata)

    def run(self):
        self.check_missing_values()
        self.check_cardinality()
        self.check_low_cardinality()
        self.check_validation_split()
        self.check_memory_issue()
        self.check_frequency()
        self.check_train_test_dtypes()
        self.check_duplicate_columns()
        self.check_target_var()
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
        self.check_low_cardinality()
        self.check_validation_split()
        self.check_class_balance()
        self.check_memory_issue()
        self.check_train_test_dtypes()
        self.check_duplicate_columns()
        self.check_target_var()
        return self.metadata, self.data_checks

    def check_class_balance(self):
        target_col = self.metadata['main_target_col']
        data = self.data[self.metadata['main_data_key']]
        classes = data[target_col].unique()
        class_num_rows = [data[data[target_col] == t].__len__() for t in classes]
        minority_class_ind = class_num_rows.index(min(class_num_rows))
        majority_class_ind = class_num_rows.index(max(class_num_rows))
        self.metadata['minority_class'] = classes[minority_class_ind]
        self.metadata['majority_class'] = classes[majority_class_ind]
        self.metadata['minority_class_%'] = int(100 * class_num_rows[minority_class_ind] / data.shape[0])
        self.metadata['majority_class_%'] = int(100 * class_num_rows[majority_class_ind] / data.shape[0])
        self.data_checks['data_checks'] = dict()
        if self.metadata['minority_class_%']/self.metadata['majority_class_%'] < 0.2:
            self.data_checks['data_checks']['class_balance_check'] = 'FAIL'
        else:
            self.data_checks['data_checks']['class_balance_check'] = 'PASS'


class UnsupervisedDataChecker(BaseDataChecker):
    def __init__(self, data, metadata):
        super().__init__(data, metadata)

    def run(self):
        # TODO add check for only 'total' type as 'train_test' might not make sense here
        self.check_missing_values()
        self.check_cardinality()
        self.check_low_cardinality()
        self.check_validation_split()
        self.check_memory_issue()
        self.check_duplicate_columns()
        return self.metadata, self.data_checks
