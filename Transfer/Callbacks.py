import logging

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import numpy as np 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AUCEvaluation(Callback):
    def __init__(self, validation_generator):
        super(Callback, self).__init__()

        self.validation_generator = validation_generator

    def on_epoch_begin(self, epoch, logs={}):
        filenames = self.validation_generator.filenames
        nb_samples = len(filenames)

        y_pred = []
        y_true = []
        for i in range(nb_samples // 32 + 1):
            print(i)
            val_X, val_y = next(self.validation_generator)
            y_pred.extend( self.model.predict(val_X) )
            y_true.extend( val_y )

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        print(y_pred.shape)
        score = roc_auc_score(y_true, y_pred)
        logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))