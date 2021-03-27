from data_curator.data_curator_logger import DataCuratorLogger
import argparse
import warnings
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def validate_filenames(filenames):
    filenames_dict = dict()

    if len(filenames) > 2:
        raise ValueError('maximum 2 files are permitted')

    if len(filenames) == 2: 
        if filenames[0] == filenames[1]:
            warnings.warn('same file names given, considering only single and as total data')
            filenames_dict['total'] = filenames[0]
        else:
            filenames_dict['train'] = filenames[0]
            filenames_dict['test'] = filenames[1]

    if len(filenames) == 1:
        filenames_dict['total'] = filenames[0]
    
    return filenames_dict


if __name__=='__main__':

    parser = argparse.ArgumentParser(prog='DataCurator')
    parser.add_argument('-v', '--verbose', help='set verbose level (default: INFO)', 
                        nargs='?', const='DEBUG', default='INFO',
                        dest="loglevel")
    
    parser.add_argument('filename', 
                        help='data file name(s). If two are given, first is treated as train data and second as test data', 
                        nargs='+')
    
    args = parser.parse_args()
    DataCuratorLogger(args.loglevel)
    datafiles = validate_filenames(args.filename)

    dataset = dict()
    for key, datafile in datafiles.items():
        # TODO add support for more file extensions
        dataset[key] = pd.read_csv(datafile)
        logger.info("reading completed: " + datafile)

