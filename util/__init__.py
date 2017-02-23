import fnmatch
import os
import sys


def get_full_filepath(filename):
    '''
    Returns full path of the first matching file inside the PYTHONPATH (recursive).
    '''
    for path in sys.path:
        for root, _dirnames, filenames in os.walk(path, followlinks=True):
            for fn in fnmatch.filter(filenames, filename):
                fullpath = os.path.join(root, fn)
                return fullpath
    raise IOError('File not found: ' + filename)


def get_files(directory, pattern):
    '''
    Returns list of files in directory matched by pattern.

    Patterns are Unix shell style:

    *       matches everything
    ?       matches any single character
    [seq]   matches any character in seq
    [!seq]  matches any char not in seq
    '''
    if os.path.exists(directory):
        return [os.path.join(directory, fn) for fn in fnmatch.filter(os.listdir(directory), pattern)]
    return []
