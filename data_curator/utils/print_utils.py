import numpy as np
import pandas as pd
import os
import json

import logging
logger = logging.getLogger(__name__)


def create_json_serializable(dict_to_json, serialize_df=False):
    append_dict = dict()
    for key, value in dict_to_json.items():
        if isinstance(value, dict):
            append_dict[key] = create_json_serializable(value, serialize_df)
        elif isinstance(value, pd.DataFrame):
            if serialize_df:
                append_dict[key] = dict()
                for ind in value.index.values:
                    append_dict[key][ind] = dict()
                    for col in value.columns:
                        append_dict[key][ind][col] = value.loc[ind, col]
            else:
                append_dict[key] = 'pandas dataframe'
        elif isinstance(value, np.integer):
            append_dict[key] = int(value)
        elif isinstance(value, np.floating):
            append_dict[key] = float(value)
        elif isinstance(value, np.ndarray):
            append_dict[key] = value.tolist()
        elif isinstance(value, np.dtype):
            if value.str.startswith('<f'):
                append_dict[key] = 'float'
            elif value.str.startswith('<i'):
                append_dict[key] = 'int'
            elif value.str.startswith('|O'):
                append_dict[key] = 'string'
            else:
                append_dict[key] = 'undefined'
        else:
            append_dict[key] = value

    return append_dict


# TODO create a logger than printer
def pretty_print(d, indent=0):
    if indent == 0:
        print('-*'*20)
    
    for key, value in d.items():
        print('\t' * indent + str(key), end=':', flush=False)
        if isinstance(value, dict):
            print('\n')
            pretty_print(value, indent+1)
            print('\n')
        
        elif isinstance(value, pd.DataFrame):
            # TODO can use tabulate for pretty printing dataframes
            # FYI tabulate is used in apache-airflow
            if value.shape[0] > 10:
                value = 'pandas dataframe'
                print('\t' * (indent+1) + str(value))
                continue
            if value.shape[1] > 10:
                print('\n')
                value = value.T
                print(value)
                continue
        
        else:
            print('\t' * (indent+1) + str(value))
            if indent == 0:
                print('\n')
    
    if indent == 0:
        print('-*'*20)


def save_json(save_dict, save_loc, save_name):
    save_path = os.path.join(save_loc, save_name)
    with open(save_path, 'w') as fp:
        json.dump(save_dict, fp, indent=6)
        logger.info('metadata saved to {0}'.format(save_path))
