import os
import warnings


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


def get_folder(file):
    return os.path.dirname(file)
