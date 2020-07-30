import os
from os.path import join 

def make_new_folder(exDir):
    i = 1
    while os.path.isdir(join(exDir, 'Ex_' + str(i))):
        i += 1

    os.mkdir(join(exDir, 'Ex_' + str(i)))
    return join(exDir, 'Ex_' + str(i))

def create_dir(dir_path):
    """
    Create directory if it does not exist
    """
    try:
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

        return 0
    except Exception as err:
        _logger.critical('Creating directories error: {0}'.format(err))
        exit(-1)