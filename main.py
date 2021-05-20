from data_curator.data_curator_logger import DataCuratorLogger
from data_curator.data_readers.data_reader import DataReader
from data_curator.data_checkers.data_checker import DataChecker
from data_curator.data_processors.data_processor import DataProcessor
from data_curator.data_processors.params.params import Params
import os
import data_curator.utils.print_utils as pu
import data_curator.utils.file_utils as fu
import json
import argparse

import logging
logger = logging.getLogger(__name__)


LOG_LEVEL_HELP = 'set verbose level (default: INFO)'
NO_TARGET_HELP = 'set that no target column exists in data, ' \
                 'useful for general data analysis and unsupervised learning. \
                 (by default: the last column is considered as target column)'
PARAMS_HELP = 'user defined parameters' \
              'optional: n_cardinality_issue_features'
FILENAME_HELP = 'data file name(s). If two are given, first is treated as train data and second as test data'
TARGET_COLUMN_HELP = 'set the column name in the data for target column.' \
                     '(by default: the last column is considered as target column unless no-target param is active)'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='DataCurator')
    parser.add_argument('-v', '--verbose', help=LOG_LEVEL_HELP, 
                        nargs='?', const='DEBUG', default='INFO', dest="loglevel")

    # TODO think of removing -nt flag and add that as target_col const
    # TODO remove both nt and target and add them to params
    parser.add_argument('-nt', '--no-target', help=NO_TARGET_HELP, 
                        nargs='?', const=1, default=0, dest="no_target")

    parser.add_argument('-tcol', '--target_col', help=TARGET_COLUMN_HELP,
                        nargs='?', const='target', default='target', dest="target_col")

    parser.add_argument('-p', '--params', help=PARAMS_HELP, default={}, nargs='?', dest="params")

    # TODO create one big params file with all useful params and file_info
    #  so that all params can be sourced from one file
    parser.add_argument('filename', help=FILENAME_HELP, nargs='+')
    
    args = parser.parse_args()
    DataCuratorLogger(args.loglevel)
    datafiles = fu.validate_filenames(args.filename)
    data_folder = fu.get_folder(list(datafiles.values())[0])

    data_reader = DataReader(datafiles, args.no_target, args.target_col)
    data, metadata = data_reader.run()
    # pu.pretty_print(data)
    # pu.pretty_print(metadata)
    
    data_checker = DataChecker(data, metadata)
    metadata, data_checks = data_checker.run()
    # pu.pretty_print(metadata)
    # pu.pretty_print(data_checks)
    metadata_to_json = pu.create_json_serializable(metadata)
    data_checks_to_json = pu.create_json_serializable(data_checks, serialize_df=True)

    pu.save_json(metadata_to_json, data_folder, 'data_checker_meta.json')
    pu.save_json(data_checks_to_json, data_folder, 'data_checker_output.json')

    PARAMS = Params().params
    data_processor = DataProcessor(data, metadata, data_checks, PARAMS)
    # metadata, data_checks = data_processor.run()
    # pretty_print(metadata)
    # pretty_print(data_checks)
