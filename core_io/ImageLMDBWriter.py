import os
import pickle

import cv2
import numpy as np

from core_io.lmdb_writer import LMDBWriter


class ImageLMDBWriter(LMDBWriter):
    """ Wrapper to the lmdb writer with image util. functions.
    """

    __meta_data__ = None        # the meta data contains image dimension and channels information
    __meta_path__ = None
    
    def __init__(self, lmdb_path, meta_path=None, earse_exist=False, auto_start=True):
        """

        Args:
            lmdb_path (str): the path to write the lmdb database
            meta_path (str): the path to write the meta data of database item.
            earse_exist (bool): erase the existing database.
            auto_start (bool): auto start the lmdb session.

        """
        super(ImageLMDBWriter, self).__init__(lmdb_path, earse_exist, auto_start)
        
        self.__meta_path__ = meta_path
        if meta_path is None:
            self.__meta_path__ = os.path.splitext(lmdb_path)[0] + '_meta.bin'
            if not os.path.exists(self.__meta_path__):
                earse_exist = True
            
        if earse_exist:
            self.__meta_data__ = dict()
        else:
            self.__meta_data__ = pickle.load(open(meta_path, "rb"))
            
    def __del__(self):
        super(ImageLMDBWriter, self).__del__()
        pickle.dump(self.__meta_data__, open(self.__meta_path__, "wb"))
        
    def insert_new_string(self, key: str, s: str):
        """
            Insert new item.

        Args:
            key (str): data
            s (str): data in byte

        """
        self.write_str(key, s)
        
    def insert_new_image_with_ds(self, img_path: str, key: str, downsample_scale=0.5):
        """
            Insert new image from disk and rescale image

        Args:
            img_path (str): the image to be loaded.
            key (str): the database key
            downsample_scale (float): the rescale factor

        """

        img = cv2.imread(img_path)
        oriH, oriW = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (int(img.shape[1]*downsample_scale),
                               int(img.shape[0]*downsample_scale)))
        newH, newW = img.shape[:2]
        self.write_array(key, img.astype(np.uint8))
        self.__meta_data__[key] = {'OriDim': (oriH, oriW), 'dim': (newH, newW)}
        
    def insert_new_image(self, image: np.ndarray, key: str, ori_dim=None):
        """
            Insert new image from disk and rescale image

        Args:
            image (array): the image array.
            key (str): the database key
            ori_dim (List): the original resolution of the image.

        """

        img = image
        if ori_dim is not None:
            ori_h, ori_w = ori_dim[0], ori_dim[1]
        else:    
            ori_h, ori_w = img.shape[:2]
        new_h, new_w = img.shape[:2]
        self.write_array(key, img.astype(np.uint8))
        self.__meta_data__[key] = {'OriDim': (ori_h, ori_w), 'dim': (new_h, new_w)}
