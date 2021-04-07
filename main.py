from data_curator.data_curator_logger import DataCuratorLogger
from data_curator.data_readers.data_reader import DataReader
from data_curator.data_checkers.data_checker import DataChecker
from data_curator.utils.print_utils import pretty_print
import argparse
import warnings
import pandas as pd

import logging
logger = logging.getLogger(__name__)


LOG_LEVEL_HELP = 'set verbose level (default: INFO)'
NO_TARGET_HELP = 'set that no target column exists in data, useful for general data analysis and unsupervised learning. \
                 (by default: the last column is considered as target column)'
# TODO can add a cmd line parameter for target column name or index
FILENAME_HELP = 'data file name(s). If two are given, first is treated as train data and second as test data'

def validate_filenames(filenames):
    filenames_dict = dict()

    if len(filenames) > 2:
        raise ValueError('maximum 2 files are permitted')

    if len(filenames) == 2: 
        if filenames[0] == filenames[1]:
            warnings.warn('same file names given, \
                           considering only single and as total data')
            filenames_dict['total'] = filenames[0]
        else:
            filenames_dict['train'] = filenames[0]
            filenames_dict['test'] = filenames[1]

    if len(filenames) == 1:
        filenames_dict['total'] = filenames[0]
    
    return filenames_dict


if __name__=='__main__':

    parser = argparse.ArgumentParser(prog='DataCurator')
    parser.add_argument('-v', '--verbose', help=LOG_LEVEL_HELP, 
                        nargs='?', const='DEBUG', default='INFO', dest="loglevel")

    parser.add_argument('-nt', '--no-target', help=NO_TARGET_HELP, 
                        nargs='?', const=1, default=0, dest="no_target")
    
    parser.add_argument('filename', help=FILENAME_HELP, nargs='+')
    
    args = parser.parse_args()
    DataCuratorLogger(args.loglevel)
    datafiles = validate_filenames(args.filename)

    data_reader = DataReader(datafiles, args.no_target)
    data, metadata = data_reader.run()
    # pretty_print(data)
    # pretty_print(metadata)
    
    data_checker = DataChecker(data, metadata)
    metadata, data_checks = data_checker.run()
    pretty_print(metadata)
    pretty_print(data_checks)
