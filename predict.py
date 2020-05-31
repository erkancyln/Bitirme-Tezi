import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.datasets.imdb import load_data
from tensorflow.keras.models import Model
import matplotlib as mtplt
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
mtplt.use('TkAgg')
food101_model = load_model('food-101/food-101.hdf5')
food101_model.save_weights("Food-101.pb")
food101_model.save("pbfile")
converter = tf.lite.TFLiteConverter.from_keras_model(food101_model)
tflite_model = converter.convert()
open("Food-101.tflite", "wb").write(tflite_model)

fruits360_model = load_model('fruits-360/Fruits-360.h5')
fruits360_model.save_weights("Fruits-360.pb")
fruits360_model.save("pbfile")
converter = tf.lite.TFLiteConverter.from_keras_model(fruits360_model)
tflite_model = converter.convert()
open("Fruits-360.tflite", "wb").write(tflite_model)
x = Input(shape=[100, 100, 3])
y_food101 = food101_model(x)
y_fruits360 = fruits360_model(x)

model = Model(inputs=x, outputs=[y_food101, y_fruits360])

#%% Predicting classes for new image
food101_train_path = "/home/erkan/Masaüstü/DataSets/food-101/train/"
fruits360_train_path="/home/erkan/Masaüstü/DataSets/fruits-360/Training"
img = load_img("cilek.jpg",target_size=(100,100))
# img to array
img = img_to_array(img)
def getAllClassNames(dir_path):
    return os.listdir(dir_path)
fruits360ClassNames = getAllClassNames(fruits360_train_path)
food101ClassNames = getAllClassNames(food101_train_path)
allClassNames=np.concatenate([food101ClassNames,fruits360ClassNames],axis=0)
num_of_classes = len(allClassNames)
DictOfClasses = {i : allClassNames[i] for i in range(0, len(allClassNames))}
print("DictOfClasses: ",DictOfClasses)
def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

#plt.imshow((img * 255).astype(np.uint8))
#img_np_array = np.expand_dims(img, axis = 0)
#img_preprocess_input = preprocess_input(img_np_array)
#predict1 = food101_model.predict(img_preprocess_input)
#predict2 = fruits360_model.predict(img_preprocess_input)
#print("predict1:",predict1)
#print("predict2:",predict2)
#predict=np.concatenate([predict1,predict2],axis=1)
#predict=np.argmax(predict,axis=1)
#print("predict",predict)
#prediction_class = fruits360_model.predict_classes(img_preprocess_input,batch_size=1)
#print("prediction_class:",prediction_class+50)
#prediction_probs = fruits360_model.predict_proba(img_preprocess_input,batch_size=1)
#print("prediction_probs:",prediction_probs)
#class_value = id_class_name(predict,DictOfClasses)
#print(class_value)
#plt.title(class_value)
#plt.show()
#plt.figure()
#%% model save h5 - pb - tflite
model.save("model.hdf5")
model.save_weights("model.pb")
model.save("pbfile")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)


plt.imshow((img * 255).astype(np.uint8))
img_np_array = np.expand_dims(img, axis = 0)
img_preprocess_input = preprocess_input(img_np_array)
predict = model.predict(img_preprocess_input)
print("predict",predict)
print("predict",predict)
#prediction_class = fruits360_model.predict_classes(img_preprocess_input,batch_size=1)
#print("prediction_class:",prediction_class+50)
#prediction_probs = model.predict_proba(img_preprocess_input,batch_size=1)
#print("prediction_probs:",prediction_probs)
class_value = id_class_name(predict,DictOfClasses)
print("class_value",class_value)
plt.title(class_value)
plt.show()
plt.figure()
