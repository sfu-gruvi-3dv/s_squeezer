import os
import pickle

import numpy as np

from core_io.lmdb_reader import LMDBModel


class ImageLMDBModel(LMDBModel):
    """ Wrapper to the lmdb reader with image util. functions.
    """

    __meta_data__ = None
    __meta_path__ = None
    
    def __init__(self, lmdb_path, meta_path=None):
        super(ImageLMDBModel, self).__init__(lmdb_path)
        
        if meta_path is None:
            meta_path = os.path.splitext(lmdb_path)[0] + '_meta.bin'
            if not os.path.exists(meta_path):
                raise Exception('[ImageLMDBModel] can not find meta info: %s' % meta_path)
        
        self.__meta_path__ = meta_path
        self.__meta_data__ = pickle.load(open(meta_path, "rb"))
        
    def __del__(self):
        super(ImageLMDBModel, self).__del__()
        
    def getImageAndDimByKey(self, key: str, channel=3) -> [np.ndarray, np.ndarray]:
        """
            Get the image by providing key from database

        Args:
            key (str): the key to retrieval image in database
            channel (int): number of channels of the image

        Returns:
            img (array): the array of image
            dim (array): the original image resolution

        """

        img = self.read_ndarray_by_key(key, dtype=np.uint8)
        
        img_meta = self.__meta_data__[key]
        ori_img_dim = img_meta['OriDim']
        img_dim = img_meta['dim']

        img = img.reshape(int(img_dim[0]), int(img_dim[1]), channel)
        
        return img, ori_img_dim
