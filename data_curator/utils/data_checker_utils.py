
def get_base_types(dtype):
    if dtype.str.startswith('<f'):
        return_val = 'float'
    elif dtype.str.startswith('<i'):
        return_val = 'int'
    elif dtype.str.startswith('|O'):
        return_val = 'string'
    else:
        return_val = 'undefined'
    return return_val
