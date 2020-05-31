import codecs
import json
import os
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mtplt
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential ,load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from glob import glob
mtplt.use('TkAgg')
# Add to path
train_path = "Training/"
test_path = "Test/"
# show image
img = load_img(train_path+"Banana/0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()
# img to array
img = img_to_array(img)
print("img.shape",img.shape)
# Number Of Classname
className = glob(train_path + '/*')
numberOfClass = len(className)
#%% CNN Model
model = Sequential()

# Convolution Layer
model.add(Conv2D(32,(3,3),input_shape = img.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

# Flatten
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))

# Dropout
model.add(Dropout(0.5))
model.add(Dense(numberOfClass)) # output
model.add(Activation("softmax"))

# Loss Fonksiyonu
model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])
# batch_size
batch_size=32
#%% Data Generation - Train - Test
train_data_generate = ImageDataGenerator(rescale= 1./255,
                   shear_range = 0.3,
                   horizontal_flip=True,
                   zoom_range = 0.3)
test_data_generate = ImageDataGenerator(rescale= 1./255)

train_generator = train_data_generate.flow_from_directory(
        train_path,
        target_size=img.shape[:2],
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")

test_generator = test_data_generate.flow_from_directory(
        test_path,
        target_size=img.shape[:2],
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")
hist = model.fit_generator(
        generator = train_generator,
        steps_per_epoch = 1600 // batch_size,
        epochs=1,
        validation_data=test_generator,
        validation_steps=800 // batch_size
        )
print("hist",hist.history)
#%% model save h5 - pb - tflite
model.save("Fruits-360.hdf5")
model.save_weights("Fruits-360.pb")
model.save("pbfile")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("Fruits-360.tflite", "wb").write(tflite_model)
food101_train_path = "/home/erkan/Masaüstü/DataSets/food-101/train/"
#%% Predicting classes for new image
def getAllClassNames(dir_path):
    return os.listdir(dir_path)
fruits360ClassNames = getAllClassNames(train_path)
food101ClassNames = getAllClassNames(food101_train_path)
print("fruits360ClassNames: ",fruits360ClassNames)
num_of_classes = len(fruits360ClassNames)
DictOfFruits360Classes = {i : fruits360ClassNames[i] for i in range(0, len(fruits360ClassNames))}
print("DictOfFruits360Classes: ",DictOfFruits360Classes)
DictOfFood101Classes = {i : food101ClassNames[i] for i in range(0, len(food101ClassNames))}
print("food101ClassNames: ",food101ClassNames)
def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value
#DictOfClasses1= np.expand_dims(DictOfClasses, axis = 0)
#print("DictOfClasses1",DictOfClasses1)
#x = [1, 2, 3]
#y = [4, 5, 6]
#z = [7, 8, 9]
#a=np.concatenate([AllClassNames,x, y, z],axis=0)

a=np.concatenate([food101ClassNames,fruits360ClassNames],axis=0)
print("a",a)
plt.imshow((img * 255).astype(np.uint8))
img_np_array = np.expand_dims(img, axis = 0)
print("img_np_array:",img_np_array)
img_preprocess_input = preprocess_input(img_np_array)
print("img_preprocess_input:",img_preprocess_input)
prediction_class = model.predict_classes(img_preprocess_input,batch_size=1)
print("prediction_class:",prediction_class)
prediction_probs = model.predict_proba(img_preprocess_input,batch_size=1)
print("prediction_probs:",prediction_probs)
class_value = id_class_name(prediction_class,DictOfFruits360Classes)
print(class_value)
plt.title(class_value)

#resim=resim1
#img  = img/255
#img = img.reshape(-1,784)
#print("img_shape: ",img.shape)
#img_np_array = np.expand_dims(img, axis = 0)
#print("img_np_array: ",img_np_array)
#img_class=model.predict_classes(img_np_array)
#print("img_class: ",img_class)
#predictions = model.predict(img_np_array)
#print("predictions: ",predictions)
#classname = img_class[0]
#print("Class: ",classname)
#print("Class name: ",className[classname-1])
#i = 0
#plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
#plot_image(i, predictions[i], img_class, img)
#print("plot_image: ",plot_image)
#plt.subplot(1,2,2)
#plot_value_array(i, predictions[i],  img_class)
#plt.show()
#labels = map(lambda x: dict(enumerate(data['target_names']))[x], data['target'])
#model = load_model('Fruits-360.hdf5')
#image_path="muz.jpg"
#resim = image.load_img(image_path, target_size=(100, 100))
#plt.imshow(resim)
#resim = np.expand_dims(resim, axis=0)
#result=model.predict_classes(resim)
#plt.title(get_label_name(result[0][0]))
#plt.show()
#%% model evaluation
print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.plot(hist.history["accuracy"], label = "Train Accuracy")
plt.plot(hist.history["val_accuracy"], label = "Validation Accuracy")
plt.legend()
plt.show()
plt.figure()

#%% save history
hist_df = pd.DataFrame(hist.history)
labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('labels.txt', 'w') as f:
    f.write(labels)
# save to json:
with open('history.json', mode='w') as f:
    hist_df.to_json(f)

# save to csv:
history_csv = 'history.csv'
with open(history_csv, mode='w') as f:
    hist_df.to_csv(f)

#%% load history
with codecs.open('load_history.json', mode='r',encoding = 'utf-8', buffering=1,errors='strict') as f:
    h = json.loads(f.read())
print(h.keys())
plt.plot(h["loss"], label = "Train Loss")
plt.plot(h["val_loss"], label = "Validation Loss")
plt.plot(h["acc"], label = "Train Accuracy")
plt.plot(h["val_acc"], label = "Validation Accuracy")
plt.legend()
plt.show()
plt.figure()


















