import os
import lmdb
import shutil

import numpy as np


class LMDBWriter(object):
    """ Write the dataset to LMDB database
    """

    ''' Variables
    '''
    __key_counts__ = 0

    # LMDB environment handle
    __lmdb_env__ = None

    # LMDB context handle
    __lmdb_txn__ = None

    # LMDB Path
    lmdb_path = None

    def __init__(self, lmdb_path, earse_exist=False, auto_start=True):
        """

        Args:
            lmdb_path (str): the path to write the lmdb database
            earse_exist (bool): erase the existing database.
            auto_start (bool): auto start the lmdb session.

        """
        self.lmdb_path = lmdb_path
        if earse_exist is True:
            self.__del_and_create__(lmdb_path)
        if auto_start is True:
            self.__start_session__()

    def __del__(self):
        self.close_session()

    @staticmethod
    def __del_and_create__(lmdb_path: str):
        """ Delete the exist lmdb database and create new lmdb database.
        """
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        os.mkdir(lmdb_path)

    def __start_session__(self):
        self.__lmdb_env__ = lmdb.Environment(self.lmdb_path, map_size=1099511627776)
        self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True)

    def close_session(self):
        if self.__lmdb_env__ is not None:
            self.__lmdb_txn__.commit()
            self.__lmdb_env__.close()
            self.__lmdb_env__ = None
            self.__lmdb_txn__ = None

    def write_str(self, key: str, ss: str):
        """
            Write the str data to the LMDB

        Args:
            key (str): key in string type
            ss (str): array data

        """
        # Put to lmdb
        self.__key_counts__ += 1
        self.__lmdb_txn__.put(key.encode(), ss)
        if self.__key_counts__ % 10000 == 0:
            self.__lmdb_txn__.commit()
            self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True, buffers=True)

    def write_array(self, key: str, array: np.ndarray):
        """
            Write the array data to the LMDB

        Args:
            key (str): key in string type
            array (array): array data

        """
        # Put to lmdb
        self.__key_counts__ += 1
        self.__lmdb_txn__.put(key.encode(), array.tostring())
        if self.__key_counts__ % 10000 == 0:
            self.__lmdb_txn__.commit()
            self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True, buffers=True)
