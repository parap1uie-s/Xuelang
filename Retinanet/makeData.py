import pandas as pd
import os
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
	# 以下两个路径需要修改
	datapath = "/home/professorsfx/Xuelang/train/"
	normal_csv = pd.read_csv( "normal.csv")
	abnormal_csv = pd.read_csv( "abnormal.csv")

	normal_csv['filename'] = datapath + "正常/" + normal_csv['filename']
	abnormal_csv['filename'] = datapath + abnormal_csv['classes'] + "/" + abnormal_csv['filename']

	train_txt = normal_csv.append(abnormal_csv)
	train_txt.loc[ train_txt['classes'] != "正常", "classes" ] = "abnormal"
	train_txt.loc[ train_txt['classes'] == "正常", "classes" ] = "normal"

	all_images = train_txt['filename'].value_counts().reset_index()
	all_images.columns = ["filename", 'counts']
	all_images = all_images.merge(train_txt[['filename','classes']], on='filename').drop_duplicates().reset_index(drop=True)

	train_image, val_image = train_test_split(all_images, test_size=0.1, random_state=42, stratify=all_images['classes'])

	train = train_image.merge(train_txt, on='filename', how='left')
	val = val_image.merge(train_txt, on='filename', how='left')

	print(train.shape)
	print(val.shape)

	train.to_csv("train_annotations.csv", columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', "classes_x"], header=False, index=False)
	val.to_csv("val_annotations.csv", columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', "classes_x"], header=False, index=False)

	f = open("classes.csv", "w+")
	# class_name,id
	f.write("{},{}\n".format("normal",0))
	f.write("{},{}\n".format("abnormal",1))
	f.close()
