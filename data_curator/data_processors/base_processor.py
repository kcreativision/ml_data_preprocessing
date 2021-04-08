import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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


class RegressionDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks):
        super().__init__(data, metadata, data_checks)


class ClassificationDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks):
        super().__init__(data, metadata, data_checks)


class UnsupervisedDataProcessor(BaseDataProcessor):
    def __init__(self, data, metadata, data_checks):
        super().__init__(data, metadata, data_checks)
