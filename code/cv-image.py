# -*- coding: utf-8 -*-
"""
Created on 5/14/2024 10:58:29

@FileName: cv-image.py
@Author: Cao Yuqi
"""
import os
import numpy as np
import cv2

def select_image_from_dir(dir_name,image_size = [224,224]):
    images = os.listdir(dir_name)
    image_num = len(images)
    select_image_index = np.random.randint(image_num)
    image_data = cv2.imread(os.path.join(dir_name,images[select_image_index]))
    image_data = cv2.resize(image_data,image_size)
    return image_data

def trainingDataSetGenerate(root_dir,output_dir,data_num):
    if not os.path.exists(output_dir):os.mkdir(output_dir)
    dirs = [os.path.join(root_dir, dir_name) for dir_name in os.listdir(root_dir)]
    for source_dir_name in dirs:
        class_name = os.path.split(source_dir_name)[-1]
        if not os.path.exists(os.path.join(output_dir,class_name)):
            os.mkdir(os.path.join(output_dir,class_name))

        for i in range(data_num):
            confusion_dir = dirs[np.random.randint(0, 9)]
            class_coeff = np.random.random(2)
            class_coeff = class_coeff/class_coeff.sum()
            confusion_coeff = np.random.random()*0.4
            class_coeff = class_coeff*(1-confusion_coeff)
            print(confusion_dir,class_coeff,confusion_coeff)
            image_data = select_image_from_dir(source_dir_name)*class_coeff[0]+\
                         select_image_from_dir(source_dir_name)*class_coeff[1]+select_image_from_dir(confusion_dir)*confusion_coeff
            cv2.imwrite(os.path.join(output_dir,class_name,str(i)+'.png'),image_data)

if __name__ == '__main__':
    root_dir = 'D:\\xihonghuaML\\xihonghuaData'
    output_dir = 'D:\\xihonghuaML\\trainingSet'
    trainingDataSetGenerate(root_dir,output_dir,50)
