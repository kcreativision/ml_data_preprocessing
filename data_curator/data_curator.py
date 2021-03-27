from data_curator_logger import DataCuratorLogger
import argparse


if __name__=='__main__':

    parser = argparse.ArgumentParser(prog='DataCurator')
    parser.add_argument('-v', '--verbose', help='set verbose level (default: INFO)', 
                        nargs='?', const='DEBUG', default='INFO',
                        dest="loglevel")
    
    parser.add_argument('filename', 
                        help='data file name(s). If two are given, first is treated as train data and second as test data', 
                        nargs='+')
    
    args = parser.parse_args()   
    print(args)
    DataCuratorLogger(args.loglevel)

