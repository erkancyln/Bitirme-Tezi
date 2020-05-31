import os
import cv2
from glob import glob
import tensorflow as tf
import matplotlib as mtplt
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.image as resim
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
mtplt.use('TkAgg')
# Add to path
train_path = "train/"
test_path = "test/"
# show image/home/erkan/Masaüstü/DataSets/pizza.jpg
img = load_img("pizza.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()
# img to array
img = img_to_array(img)
# Number Of Classname
className = glob(train_path + '/*')
numberOfClass = len(className)

os.listdir('/home/erkan/Masaüstü/DataSets/food-101/food-101/images')
os.listdir('/home/erkan/Masaüstü/DataSets/food-101/food-101/meta')
#def prepare_data(filepath, src,dest):
#    classes_images = defaultdict(list)
#    with open(filepath, 'r') as txt:
#        paths = [read.strip() for read in txt.readlines()]
#        for p in paths:
#            food = p.split('/')
#            classes_images[food[0]].append(food[1] + '.jpg')
#
#    for food in classes_images.keys():
#        print("\nCopying images into ",food)
#        if not os.path.exists(os.path.join(dest,food)):
#            os.makedirs(os.path.join(dest,food))
#        for i in classes_images[food]:
#            copy(os.path.join(src,food,i), os.path.join(dest,food,i))
#    print("Copying Done!")
# Prepare train dataset by copying images from food-101/images to food-101/train using the file train.txt
#print("Creating train data...")
#prepare_data('/home/erkan/Masaüstü/DataSets/food-101/food-101/meta/train.txt',
#             '/home/erkan/Masaüstü/DataSets/food-101/food-101/images', 'train')
# Prepare test data by copying images from food-101/images to food-101/test using the file test.txt
#print("Creating test data...")
#prepare_data('/home/erkan/Masaüstü/DataSets/food-101/food-101/meta/test.txt',
#             '/home/erkan/Masaüstü/DataSets/food-101/food-101/images', 'test')
# Helper method to create train_mini and test_mini data samples
#def dataset_mini(food_list, src, dest):
#    if os.path.exists(dest):
#        rmtree(dest) # removing dataset_mini(if it already exists) folders so that we will have only the classes that we want
#        os.makedirs(dest)
#    for food_item in food_list :
#        print("Copying images into",food_item)
#        copytree(os.path.join(src,food_item), os.path.join(dest,food_item))

# picking 10 of my favorite foods
food_list = ['nachos','ice_cream','sushi','french_fries','bibimbap','cheesecake','donuts','dumplings','waffles','omelette']
src_train = 'train'
dest_train = 'train_mini'
src_test = 'test'
dest_test = 'test_mini'

#print("Creating train data folder with new classes")
#dataset_mini(food_list, src_train, dest_train)
#
#print("Creating test data folder with new classes")
#dataset_mini(food_list, src_test, dest_test)


tf.compat.v1.disable_eager_execution()
K.clear_session()
train_dir = 'train_mini'
test_dir = 'test_mini'
train_dir_2 = '/home/erkan/Masaüstü/DataSets/food5k/Food-5K/training'
test_dir_2 = '/home/erkan/Masaüstü/DataSets/food5k/Food-5K/validation'
img_height = 224
img_width = 224
batch_size = 16
num_classes = 10
num_train_samples = 7500
num_test_samples = 2500

#%% Data Augmentation - Train - Validation - Test
# Training datagen and generator
# image data generators for image inputs

#train_imgen = ImageDataGenerator(rescale=1. / 255,
#                                         shear_range=0.2,
#                                         zoom_range=0.2,
#                                         rotation_range=40,
#                                         width_shift_range=0.2,
#                                         height_shift_range=0.2,
#                                         horizontal_flip=True,
#                                         validation_split=0.1,
#                                         fill_mode='nearest')
#
#
#val_imgen = ImageDataGenerator(
#    rotation_range=40,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    rescale=1./255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    validation_split=0.1,
#    fill_mode='nearest')


#def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
#    genX1 = generator.flow_from_directory(dir1,
#                                          target_size=(img_height, img_width),
#                                          class_mode='categorical',
#                                          batch_size=batch_size,
#                                          subset='training',
#                                          shuffle=False,
#                                          seed=7)

#    genX2 = generator.flow_from_directory(dir2,
#                                          target_size=(img_height, img_width),
#                                          class_mode='categorical',
#                                          batch_size=batch_size,
#                                          shuffle=False,
#                                          seed=7)
#    while True:
#        X1i = genX1.next()
#        X2i = genX2.next()
#        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label


#train_generator = generate_generator_multiple(generator=train_imgen,
#                                             dir1=train_dir,
#                                             dir2=train_dir_2,
#                                             batch_size=batch_size,
#                                             img_height=img_height,
#                                             img_width=img_height)
#
#val_generator = generate_generator_multiple(val_imgen,
#                                            dir1=train_dir,
#                                            dir2=train_dir_2,
#                                            batch_size=batch_size,
#                                            img_height=img_height,
#                                            img_width=img_height)
#
#dataset = tf.data.Dataset.from_tensor_slices([img.shape])
#print("dataset",dataset)
#def concat_datasets(datasets):
#    ds0 = datasets.from_tensors(datasets[0])
#    for ds1 in datasets[1:]:
#        ds0 = ds0.concatenate(datasets.from_tensors(ds1))
#    return ds0

#ds = tf.data.Dataset.zip(tuple(datasets)).flat_map(
#    lambda *args: concat_datasets(args)
#)
train_data_generate = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1,
    fill_mode='nearest')

train_gen = train_data_generate.flow_from_directory(
    train_dir,
    target_size = (img_height,img_width),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training')

# Validation datagen and generator
val_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1,
    fill_mode='nearest')

val_gen = train_data_generate.flow_from_directory(
    train_dir,
    target_size = (img_height,img_width),
    batch_size = 1,
    class_mode = 'categorical',
    subset = 'validation',
    shuffle = False)

# Testing datagen and generator
test_datagen = ImageDataGenerator()

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size = (img_height,img_width),
    batch_size = 1,
    class_mode = 'categorical',
    shuffle = False)
#%% Building Model Architecture
resnet50 = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)
resnet50.trainable = False
x = resnet50.output
x = GlobalAveragePooling2D()(x)
prediction = Dense(num_classes, activation='softmax')(x)

model = Model(inputs = resnet50.input, outputs = prediction)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(train_gen,
                    steps_per_epoch = train_gen.samples // batch_size//84,
                    epochs=1,
                    validation_data = val_gen,
                    validation_steps = val_gen.samples)
#history=model.fit_generator(train_generator,
#                        steps_per_epoch=422/batch_size,
#                        epochs = 1,
#                        validation_data = val_generator,
#                        validation_steps = 750,
#                        use_multiprocessing=True,
#                        shuffle=False)
# Save model weights
model.save('transfer_learning_10class.hdf5') #Model with presaved ImageNet weights and only training the fully connected layers

history.history.keys();
plt.figure(figsize = (10,5))
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.legend(['val_accuracy','train_accuracy'])
plt.show()
# Load the pre-trained model if it is not already
# from tensorflow.keras.models import load_model
# model = load_model(transfer_learning_10class.hdf5)

#The last 17 layers as trainable (which is the last convolutional block + the fully connected layers)
for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-17:]:
    layer.trainable = True

optim = optimizers.SGD(learning_rate=1e-4, momentum=0.9)
model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])
model.summary()
#%% Predicting classes for new image
def getAllClassNames(dir_path):
    return os.listdir(dir_path)
AllClassNames = getAllClassNames(train_path)
num_of_classes = len(AllClassNames)
DictOfClasses = {i : AllClassNames[i] for i in range(0, len(AllClassNames))}
print("DictOfClasses: ",DictOfClasses)
def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

plt.imshow((img * 255).astype(np.uint8))
img_np_array = np.expand_dims(img, axis = 0)
img_preprocess_input = preprocess_input(img_np_array)
predict = model.predict(img_preprocess_input)
print("predict",predict)
predict=np.argmax(predict,axis=1)
class_value = id_class_name(predict,DictOfClasses)
print("class_value",class_value)
plt.title(class_value)
plt.show()
plt.figure()

#%% model save h5 - pb - tflite
model.save("food-101.h5")
model.save("food-101.pb")
model.save("pbfile")
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()
#open("food-101.tflite", "wb").write(tflite_model)
#plt.figure(figsize=(10,5))
#plt.plot(history.history['val_accuracy'] + history.history['val_accuracy'])
#plt.plot(history.history['accuracy'] + history.history['accuracy'])
#plt.plot([history.epoch[-1],history.epoch[-1]],
#         plt.ylim(), label='Start Fine Tuning')
#plt.legend(['val_accuracy','train_accuracy'])
#plt.show()


