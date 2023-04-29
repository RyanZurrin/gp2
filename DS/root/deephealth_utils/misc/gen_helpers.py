# import sys


class MultWriter(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)


def is_str_type(val):
    # if sys.version[0] == '2':
    #     return type(val) in [str, unicode]
    # else:
    #     return isinstance(val, str)
    
    # we just assume centaur will be ran in python3
    return isinstance(val, str)
