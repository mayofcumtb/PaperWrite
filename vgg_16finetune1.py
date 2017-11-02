'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
fine-tuning the weight parameters using pre-trained network
'''
import sys
sys.path.append("/data/dongdong.yu/dllibs/")

import os
import h5py
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
# path to the model weights files.

def predict_image(image_path, model):
	img = image.load_img(image_path, target_size=(150,150))
	x = image.img_to_array(img)
	x = np.extend_dims(x,axis=0)
	x = pre

weights_path = 'vgg16_weights.h5'
# top_model_weights_path = 'fc_model.h5'
# top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/data/zairan.wang/YanMA/data/syn_data/train'
validation_data_dir = '/data/zairan.wang/YanMA/data/objectnet3d/validation'
test_data_dir = '/data/zairan.wang/YanMA/data/objectnet3d/test'
syn_test_data_dir = '/data/zairan.wang/YanMA/data/syn_data/test'
nb_train_samples = 19531
nb_validation_samples = 25303
nb_epoch = 20

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))

model.add(Conv2D(64, (3, 3), activation="relu", name="conv1_1"))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation="relu", name="conv1_2"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation="relu", name="conv2_1"))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
need_back = True
if need_back == True:

	assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
	f = h5py.File(weights_path)
	for k in range(f.attrs['nb_layers']):
		if k >= len(model.layers):
			# we don't look at the last (fully-connected) layers in the savefile
			break
		g = f['layer_{}'.format(k)]
		# kernel = g['param_{}'.format(0)]
		# bias = g['param_{}'.format(1)]
		# kernel = np.transpose(kernel,(2,3,1,0))
		weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
		if len(weights):
			kernel = weights[0]
			bias = weights[1]
			kernel = np.transpose(kernel, (2, 3, 1, 0))
			weights = [kernel, bias]
		# weights = [kernel, bias]

		model.layers[k].set_weights(weights)
	f.close()
	print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

if need_back == False:
	final_weights_path = 'my_model_weights_synthetic.h5'
	model.load_weights(final_weights_path)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
	layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
			  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
			  metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0,
	zoom_range=0,
	horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_height, img_width),
	batch_size=64,
	class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_height, img_width),
	batch_size=64,
	class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
	test_data_dir,
	target_size=(img_height, img_width),
	batch_size=64,
	class_mode='categorical'
)

syn_test_generator = test_datagen.flow_from_directory(
	syn_test_data_dir,
	target_size=(img_height, img_width),
	batch_size=64,
	class_mode='categorical'
)

# fine-tune the model
if need_back:
	model.fit_generator(
		train_generator,
		samples_per_epoch=nb_train_samples,
		nb_epoch=nb_epoch,
		validation_data=validation_generator,
		nb_val_samples=nb_validation_samples)
else:
	pass
predict_single_image = False
if predict_single_image:
	pass
else:
	print 'evaluate validation'
	score = model.evaluate_generator(validation_generator,25303)
	print 'validation score is {}'.format(score[1])
	print 'evaluate test'
	score = model.evaluate_generator(test_generator,6510)
	print 'test score is {}'.format(score[1])
	print 'evaluate syntest'
	score = model.evaluate_generator(syn_test_generator,24097)
	print 'synthetic test score is {}'.format(score[1])
	model.save_weights('my_model_weights_synthetic.h5')

each_categories = os.path.join("/data/zairan.wang/YanMA/data/syn_data/each_category/")
each_categorie_files = os.listdir(each_categories)
for files in each_categorie_files:
	tmp_dir = os.path.join(each_categories,files)
	if not os.path.isdir(tmp_dir):
		continue
	# pdb.set_trace()
	tmp_test_generator = test_datagen.flow_from_directory(
		tmp_dir,
		target_size=(img_height, img_width),
		batch_size=64,
		classes=sort_classes,
		class_mode='categorical'
	)
	score = model.evaluate_generator(tmp_test_generator,200)
	print 'synthetic category {} test score is {}'.format(files ,score[1])

# preds = model.predict_generator(test_generator, 8139)
# pdb.set_trace()
each_categories = os.path.join("/data/zairan.wang/YanMA/data/objectnet3d/each_category/")
each_categorie_files = os.listdir(each_categories)
for files in each_categorie_files:
	tmp_dir = os.path.join(each_categories,files)
	if not os.path.isdir(tmp_dir):
		continue
	# pdb.set_trace()
	tmp_test_generator = test_datagen.flow_from_directory(
		tmp_dir,
		target_size=(img_height, img_width),
		batch_size=64,
		classes=sort_classes,
		class_mode='categorical'
	)
	score = model.evaluate_generator(tmp_test_generator,200)
	print 'objectnet category {} test score is {}'.format(files ,score[1])