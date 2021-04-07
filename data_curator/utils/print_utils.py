import pandas as pd

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
