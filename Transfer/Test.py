import pandas as pd
from PIL import Image
import numpy as np
import os
from Model import *
from Utils import TestDataGen
from keras.preprocessing.image import ImageDataGenerator
import argparse

if __name__ == '__main__':
    datapath = "dataset/"
    width = 600
    height = 600
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelType', required=True)
    parser.add_argument('--channel', required=False, default="4")
    args = parser.parse_args()

    assert args.modelType in ["densenet", "InceptionResNetV2", "Resnet", "xception", "inception", "nas"]
    assert args.channel.isdigit()
    args.channel = int(args.channel)

    if args.modelType == "densenet":
        model = DenseNetTransfer((height,width,3), channel=args.channel)
    elif args.modelType == "InceptionResNetV2":
        model = Transfer((height,width,3), channel=args.channel)
    elif args.modelType == "Resnet":
        model = ResNet((height,width,3), channel=args.channel)
    elif args.modelType == "xception":
        model = XceptionTransfer((height,width,3), channel=args.channel)
    elif args.modelType == "inception":
        model = InceptionTransfer((height,width,3), channel=args.channel)
    elif args.modelType == "nas":
        model = NASTransfer((height,width,3), channel=args.channel)
        
    model.load_weights("Transfer-{}.h5".format(args.modelType), by_name=True)

    class_indices = {'abnormal': 0, 'normal': 1}
    class_indices = dict((v,k) for k,v in class_indices.items())
    result = {}

    test_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
            os.path.join(datapath, "test"),
            target_size=(height, width),
            shuffle = False,
            class_mode=None,
            batch_size=1,
            follow_links=True)

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    res = model.predict_generator(test_generator,steps = nb_samples)
    print(res.shape)
    for k,file in enumerate(filenames):
        result[file.split("/")[1]] = res[k][0]

    pred_result = pd.DataFrame.from_dict(result,orient='index').reset_index()
    pred_result.columns = ['filename', 'probability']
    pred_result.loc[ pred_result['probability'] >= 1.0, "probability" ] = 1 - 1e-7
    
    pred_result.to_csv("result-{}.csv".format(args.modelType),index=False)