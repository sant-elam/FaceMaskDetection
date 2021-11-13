# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 00:01:43 2021

@author: 
"""
'''
 FACE DETECTION
 
'''

import mediapipe as mp
import cv2 as cv

import os 
import glob
import numpy as np

from sklearn.model_selection import train_test_split


from keras.utils import np_utils

import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


import tensorflow as tf

def check_for_gpu(gpu_index=0):
    gpu_config = tf.config
    gpus = gpu_config.list_physical_devices('GPU')

    if gpus:
        # Restrict TensorFlow to only allocate 3GB of memory on the first GPU

        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

            gpu_config.set_logical_device_configuration(gpus[0],
                                                        [gpu_config.LogicalDeviceConfiguration(memory_limit=5120)])

            logical_gpus = gpu_config.list_logical_devices('GPU')

            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
            

def create_sequential(no_of_classes, target_size=28):

    model = Sequential()
    
    # FIRST LAYER
    model.add(Conv2D(32, (3,3), activation = 'relu', input_shape=(target_size, target_size, 3)))
    model.add(MaxPooling2D())  
    
    # SECOND LAYER
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D())  
    
    # THIRD LAYER
    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D())  
    
    
    model.add(Flatten())  
    
    model.add(Dense(512, activation='relu'))
              
    model.add(Dense(no_of_classes, activation='softmax'))
    
    model.compile( optimizer='adam',
                   loss = 'categorical_crossentropy',
                   metrics = 'accuracy')
    
    return model


def draw_with_mediapipe(image, outputs, draw_color):
    for region in outputs.detections:
        draw_utils.draw_detection (image,
                                   region,
                                   draw_color,
                                   draw_color)
             
    
def get_keypoints(image, key_points_pos, region, color, radius):
    
    height, width = image.shape[0:2]
     
    left_eye = region.location_data.relative_keypoints[key_points_pos]
    
    x_point = left_eye.x                 
    y_point = left_eye.y
    
    x_point = (int)(x_point * width)
    y_point = (int)(y_point * height)
    
    if x_point <0:
        x_point =0
    if y_point <0:
        y_point = 0
    
    cv.circle(image, (x_point, y_point), radius, color, 1)
    
    return x_point, y_point

def get_bounding_rectangle(image,region, color, radius):
    
    height, width = image.shape[0:2]
    
    bounding_box = region.location_data.relative_bounding_box   
                 
    left = bounding_box.xmin
    top = bounding_box.ymin
    
    right = left + bounding_box.width
    bottom = top + bounding_box.height
    
    left = (int)(left * width)
    top = (int)(top * height)
    
    if left <0:
        left =0
    if top <0:
        top = 0
    
    right = (int)(right * width)
    bottom = (int)(bottom * height)
    
    cv.rectangle(image, (left, top), (right, bottom), COLOR_RED, 1)
    
    cv.line(image, (left, top), (left, top+10), COLOR_RED, 3)
    cv.line(image, (left, top), (left+10, top), COLOR_RED, 3)
    
    cv.line(image, (right, top), (right, top+10), COLOR_RED, 3)
    cv.line(image, (right, top), (right-10, top), COLOR_RED, 3)
    
    
    cv.line(image, (left, bottom), (left, bottom-10), COLOR_RED, 3)
    cv.line(image, (left, bottom), (left+10, bottom), COLOR_RED, 3)
    
    cv.line(image, (right, bottom), (right, bottom-10), COLOR_RED, 3)
    cv.line(image, (right, bottom), (right-10, bottom), COLOR_RED, 3)
       
    
    return left, top, right, bottom


def read_image_resize(dataset_folder, target_size, detection_module):
    
    images =[]
    labels =[]
    for directory in os.listdir(dataset_folder):
        #print(directory)
        
        path_dir = os.path.join(dataset_folder, directory)
        
        all_images = glob.glob(path_dir + "**/*.jpg")
        
        for image_file in all_images:
            
            
            image = cv.imread(image_file)
           
            image_convert = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            outputs = detection_module.process(image_convert)
            
            if outputs.detections:
    
                 for region in outputs.detections:   
                     
                    
                    left, top, right, bottom = get_bounding_rectangle(image,region, COLOR_GREEN, 2)
                    
                    crop_image = image[top:bottom, left:right]
                    
                    #print(left, top, right, bottom)
                    
                    #draw_with_mediapipe(image, outputs, draw_color)
                    #left_eye_x, left_eye_y = get_keypoints(image, LEFT_EYE, region, COLOR_GREEN, 20)
                    #right_eye_x, right_eye_y = get_keypoints(image, RIGHT_EYE, region, COLOR_GREEN, 20)                    
                                                 
                    
                    height, width = crop_image.shape[0:2]
                    
                            
                    image_resize = cv.resize(crop_image, (target_size, target_size))
                    
                    h,w = image_resize.shape[0:2]
                    
                    print (height, width)  
                    
                    if width <50 or height<50:
                        cv.imshow("Padd_image", image_resize)          
                        cv.waitKey(0)
                        
                        cv.imshow("Padd_image", image) 
           
                        cv.waitKey(0)
                    else:
                        images.append(image_resize)
                    
                        labels.append(directory)
                  
    return images, labels


def split_train_test(images, labels, testsize= 0.33, target_size=256, randomstate=1):
    
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size= testsize, random_state = randomstate)
           
     
    #2.4 Convert the images/dataset to numpy array
    x_train = np.array(x_train, dtype=object)    
    x_test = np.array(x_test, dtype=object)
        
    #2.5 Reshape the array
    no_of_train = x_train.shape[0]     
    no_of_test  = x_test.shape[0]
    
    print(x_train.shape)
    print(x_test.shape)
    
    x_train = x_train.reshape(no_of_train, target_size, target_size, 3)
    x_test  = x_test.reshape(no_of_test, target_size, target_size, 3)
    
    #print(x_train[0])
         
    #2.6 Convert array to float
    x_train = x_train.astype('float32')     
    x_test = x_test.astype('float32')
    
    
    #print(x_train[0])
         
    #2.7 Noramlized or scale - 0:1
         
    x_train /= 255
    x_test /= 255
    
    return x_train, x_test, y_train, y_test

def one_hot_encode(y_train, y_test, label_array, no_of_classes):
    y_train_keys=[]
    for y_label in y_train:
        values =    label_array[y_label]  
        y_train_keys.append(values)

    y_test_keys=[]
    for y_label in y_test:
        values =    label_array[y_label]  
        y_test_keys.append(values)
        
    y_train = np_utils.to_categorical(y_train_keys, no_of_classes)
    y_test = np_utils.to_categorical(y_test_keys, no_of_classes)
        
    return y_train, y_test

RIGHT_EYE = 0
LEFT_EYE = 1
NOSE_TIP = 2
MOUTH_CENTER = 3
RIGHT_EAR_TRAGION = 4
LEFT_EAR_TRAGION = 5

COLOR_GREEN = (0,255,0)
COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)

check_for_gpu(0)

face_detection = mp.solutions.face_detection
draw_utils = mp.solutions.drawing_utils
draw_color = draw_utils.DrawingSpec((0,255,0), thickness=3, circle_radius = 1)


#video_path = "C:/Users/Santosh/OneDrive/Desktop/FACE_DETECTION/VID-20211028-WA0001.mp4"

#capture = cv.VideoCapture(0)


MODEL_SELECTION = 1
CONFIDENCE = 0.5

TARGET_SIZE = 256

detection = face_detection.FaceDetection(model_selection=MODEL_SELECTION, 
                                         min_detection_confidence= CONFIDENCE)

dataset_folder = "C:/MEDIA_PIPE/FACE_MASK/face-mask-dataset/Dataset/train/"


images, labels = read_image_resize(dataset_folder, TARGET_SIZE, detection)


#2.4 Split the images into Train set and Test set
x_train, x_test, y_train, y_test = split_train_test(images, 
                                                    labels, 
                                                    0.33, 
                                                    TARGET_SIZE, 
                                                    2)

print(labels)
NO_OF_CLASSES = 2     
label_array = {'with_mask':0, 'without_mask':1}


y_train, y_test = one_hot_encode(y_train, 
                                 y_test, 
                                 label_array, 
                                 NO_OF_CLASSES)


# 4. SIMPLE MODEL TO TRAIN..
     # 3 layes models... 

model = create_sequential(NO_OF_CLASSES, TARGET_SIZE)

model.summary()


Epochs_to_train = 25

# 5. TRAIN the dataset with this MODEL..
model.fit(x_train, y_train,
          validation_data = ( x_test, y_test),
          epochs = Epochs_to_train)

# 6. PLOT and see how the model performs on this data
print(model.history.history)

history = model.history.history

plt.subplot(2, 1, 1)
plt.title("MODEL LOSS")
plt.ylabel("Loss")
plt.xlabel("No of Epochs")

plt.plot(history['loss'])
plt.plot(history['val_loss'])

plt.legend(['train', 'test'], loc='upper right')


plt.subplot(2, 1, 2)
plt.title("MODEL Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("No of Epochs")

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])

plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()



save_model_path = "C:/MEDIA_PIPE/FACE_MASK/FaceMaskGPU.h5"

model.save(save_model_path)


# For test purpose
capture = cv.VideoCapture(0)
while True:
    results, image = capture.read()
    
    if results:
        image_convert = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        outputs = detection.process(image_convert)

        if outputs.detections:

             for region in outputs.detections:   
                 
                
                left, top, right, bottom = get_bounding_rectangle(image,region, COLOR_GREEN, 2)
                
                crop_image = image[top:bottom, left:right]
                
                #print(left, top, right, bottom)
                
                #draw_with_mediapipe(image, outputs, draw_color)
                #left_eye_x, left_eye_y = get_keypoints(image, LEFT_EYE, region, COLOR_GREEN, 20)
                #right_eye_x, right_eye_y = get_keypoints(image, RIGHT_EYE, region, COLOR_GREEN, 20)                    
                                             
                
                height, width = crop_image.shape[0:2]
        
                    
                image_resize = cv.resize(crop_image, (TARGET_SIZE, TARGET_SIZE))
                
                
                image_s = np.array(image_resize, dtype=object)    
 
                image_s = image_s.reshape(1, TARGET_SIZE, TARGET_SIZE, 3)

                image_s = image_s.astype('float32')     

                image_s /= 255
              
                detect_class = model.predict_classes(image_s)
                
                print(detect_class)

                cv.imshow("Face_Detection", image)     
                
                if cv.waitKey(1) & 255 == 27:
                    break
        
capture.release()
cv.destroyAllWindows()
'''
for directory in os.listdir(dataset_folder):
        print(directory)
        
        path_dir = os.path.join(dataset_folder, directory)
        
        print(path_dir)
        
        all_images = glob.glob(path_dir + "**/*.jpg")
        
        for image_path in all_images:
            
            print(image_path)
            image = cv.imread(image_path)
           
            image_convert = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            height, width = image_convert.shape[0:2]
            
            outputs = detection.process(image_convert)
            
            if outputs.detections:
    
                 for region in outputs.detections:   
                     
                     #draw_with_mediapipe(image, outputs, draw_color)
                     left, top, right, bottom = get_bounding_rectangle(image,region, COLOR_GREEN, 2)
                     
                     crop_image = image[top:bottom, left:right]
                     
                     print(left, top, right, bottom)
                     
                     #left_eye_x, left_eye_y = get_keypoints(image, LEFT_EYE, region, COLOR_GREEN, 20)
                     #right_eye_x, right_eye_y = get_keypoints(image, RIGHT_EYE, region, COLOR_GREEN, 20)                    
                                                  
                     
            cv.imshow("Face_Detection", image) 
            
    
            cv.waitKey(0)
            
'''            
            
'''

'''