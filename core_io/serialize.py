import pickle


def dump_pickle(obj, file_path: str, verbose=False):
    """ Save object to pickle binary file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
        if verbose:
            print('[Serialization] Saved to %s' % file_path)


def load_pickle(file_path: str, verbose=False):
    """ Load object from pickle binary file
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
        if verbose:
            print('[Serialization] Load from %s' % file_path)
        return obj
