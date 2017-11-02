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
# import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
import keras

import pdb


# path to the model weights files.

weights_path = 'vgg16_weights.h5'
# top_model_weights_path = 'fc_model.h5'
# top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/data/zairan.wang/YanMA/data/objectnet3d/train'
validation_data_dir = '/data/zairan.wang/YanMA/data/syn_data/validation'
test_data_dir = '/data/zairan.wang/YanMA/data/objectnet3d/test'
syn_test_data_dir = '/data/zairan.wang/YanMA/data/syn_data/test'
nb_train_samples = 2000
nb_validation_samples = 200
nb_epoch = 10
def convert_input(img):
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)

	x = preprocess_input(x)



def predict(model, img, classes,target_size, top_n=3):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (width, height) tuple
    top_n: # of top predictions to return
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  x = x/255.0
  preds = model.predict(x)
  return decode_predictions(preds, classes,top=top_n)[0]

def decode_predictions(preds, classes,top=5):
    # pdb.set_trace()
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(classes[i],(pred[i],),pred) for i in top_indices]
        # pdb.set_trace()
        # result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def test_images(dir_path,model):
	classes = os.listdir(dir_path)
	for class_name in classes:
		image_root_file = os.path.join(dir_path, class_name)
		image_paths = os.listdir(image_root_file)
		count = 0
		total = 0
		flag = True
		# pdb.set_trace()
		for image_path in image_paths:
			image_abs_path = os.path.join(image_root_file, image_path)
			img = image.load_img(image_abs_path)
			target_size = (150,150)
			sort_classes = sorted(classes)

			result = predict(model,img,sort_classes,target_size,3)
			# print result
			# print "{} dir predict result is {}".format(class_name, result)
			# print class_name, result[0]

			if class_name in result[0]:
				count = count + 1
			total = total + 1
			if flag:
				pass
				# pdb.set_trace()
		# pdb.set_trace()
		print "{} dir predict result is {}%".format(class_name, 100*count/total)
		print count,total
			# pdb.set_trace

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
classes = os.listdir(train_data_dir)
sort_classes = sorted(classes)

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
need_back = False
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
	# final_weights_path = './best_weights.hdf5'
	final_weights_path = 'my_model_weights.h5'
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
	batch_size=512,
    classes = sort_classes,
	class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_height, img_width),
	batch_size=128,
	classes = sort_classes,
	class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
	test_data_dir,
	target_size=(img_height, img_width),
	batch_size=128,
	classes = sort_classes,
	class_mode='categorical'
)

syn_test_generator = test_datagen.flow_from_directory(
	syn_test_data_dir,
	target_size=(img_height, img_width),
	batch_size=128,
	classes = sort_classes,
	class_mode='categorical'
)
if need_back:
	best_weights_filepath = './best_weights.hdf5'
	earlyStopping= keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
	saveBestModel = keras.callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
	keras.callbacks.RemoteMonitor(root='http://0.0.0.0:12345', path='./', field='data', headers=None)
# train model
# history = model.fit(x_tr, y_tr, batch_size=batch_size, nb_epoch=n_epochs,
#               verbose=1, validation_data=(x_va, y_va), callbacks=[earlyStopping, saveBestModel])
# fine-tune the model
if need_back:
	history = model.fit_generator(
		train_generator,
		samples_per_epoch=nb_train_samples,
		steps_per_epoch=10,
		nb_epoch=100,
		workers = 12,
		validation_data=validation_generator,
		nb_val_samples=1000,
		callbacks=[earlyStopping, saveBestModel]
	)
else:
	pass
print 'evaluate validation'
score = model.evaluate_generator(validation_generator,79908)
print 'validation score is {}'.format(score[1])
print 'evaluate test'
score = model.evaluate_generator(test_generator,8139)
print 'test score is {}'.format(score[1])
print 'evaluate syntest'
score = model.evaluate_generator(syn_test_generator,79908)
print 'synthetic test score is {}'.format(score[1])
#model.save_weights('my_model_weights_11_01.h5')
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