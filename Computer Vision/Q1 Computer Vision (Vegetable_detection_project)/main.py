import cv2
import numpy as np
import tensorflow as tf




model=tf.keras.models.load_model('model/1/')
CLASS_NAMES=['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

def prediction(path):
    image=cv2.imread(path)
    image=image/255
    image=np.expand_dims(image,axis=0)
    return CLASS_NAMES[np.argmax(model.predict(image))]


if __name__=="__main__":
    print(prediction(r'C:\Users\DELL\Downloads\Inueron assignment\Deep Learning\Vegetable_detection_project\Vegetable Images\test\Bottle_Gourd\1003.jpg'))