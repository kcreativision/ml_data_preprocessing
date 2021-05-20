import logging
logger = logging.getLogger(__name__)


def basic_params_defaults(key):
    defaults = {
        'numeric_imputation_method': 'mean',
        'categorical_imputation_method': 'most_frequent',
    }
    return defaults[key]


def basic_params_options(key):
    options = {
        'numeric_imputation_method': ['mean', 'median', 'most_frequent'],
        'categorical_imputation_method': ['most_frequent'],
    }
    return options[key]


class Params(object):
    def __init__(self, user_params=None):
        if user_params is None:
            user_params = {}

        self.user_params = user_params
        self.params = {
            "numeric_imputation_method": self.get_numeric_imputation_method(),
            "categorical_imputation_method": self.get_categorical_imputation_method(),
        }

    def get_numeric_imputation_method(self):
        param_name = 'numeric_imputation_method'
        if not self.is_valid_param_optional(param_name):
            self.user_params[param_name] = basic_params_defaults(param_name)
            logger.info('using default {0}: {1}'.format(param_name, self.user_params[param_name]))
        else:
            logger.debug('{0} read successfully: {1}'.format(param_name, self.user_params[param_name]))

        return self.user_params[param_name]

    def get_categorical_imputation_method(self):
        param_name = 'categorical_imputation_method'
        if not self.is_valid_param_optional(param_name):
            self.user_params[param_name] = basic_params_defaults(param_name)
            logger.info('using default {0}: {1}'.format(param_name, self.user_params[param_name]))
        else:
            logger.debug('{0} read successfully: {1}'.format(param_name, self.user_params[param_name]))

        return self.user_params[param_name]

    def is_valid_param_compulsory(self, key):
        if key not in self.user_params.keys():
            logger.error('{0} not present in user params'.format(key))
            return 0
        else:
            return 1

    def is_valid_param_optional(self, key):
        if key not in self.user_params.keys():
            logger.debug('{0} not present in user params'.format(key))
            return 0
        elif self.user_params[key] not in basic_params_options(self.user_params[key]):
            logger.error('{0} not a valid value. '
                         'it should be one of {1}'.format(key, basic_params_options(self.user_params[key])))
            return 0
        else:
            return 1
