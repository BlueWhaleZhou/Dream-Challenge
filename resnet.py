from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Model
from keras.layers import Activation, merge
from keras.layers import UpSampling2D, Convolution2D, MaxPooling2D
from keras.layers import Input, BatchNormalization
import matplotlib.pyplot as plt
import keras.backend as K
import pandas as pd
import pydicom as dm


# processing dcm data
def reshaping(x, shape):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] == 4095:
                x[i][j] = 0
    if x.shape is not shape:
        zeros = np.zeros(shape)
        zeros[:x.shape[0], :x.shape[1]] = x
        return zeros
    else:
        return x
shape = (4096, 3328)
file_list = []
label = np.zeros(500)
path = '/home/qinghai/research/dream/pilot_images/'
label_file = '/home/qinghai/research/dream/dream/dreamchallenges/images_label.csv'
label_file_data = pd.read_csv(label_file, sep='\t')
for i in range(len(label_file_data)):
    file_list.append(label_file_data.values[i][5])
    label[i] += label_file_data.values[i][6]
print (file_list)
print (label)
filename = path + file_list[0]
data_sample = dm.read_file(filename)
data_matrix = reshaping(data_sample.pixel_array, shape)
print (data_matrix.shape)

for i in range(1, 500):
    sample_temp = dm.read_file(path + file_list[i])
    print (file_list[i])
    data_temp = sample_temp.pixel_array
    print (data_temp.shape)
    data_temp_p = reshaping(data_temp, shape)
    data_matrix = np.concatenate((data_matrix, data_temp))
data_matrix_f = data_matrix.reshape(500, 1, shape[0], shape[1])
print (data_matrix_f.shape)
np.savetxt('/home/qinghai/research/dream/dream.txt', data_matrix_f, delimiter=",")

def convresblock(x, nfeats=128, ksize=3, nskipped=2):
    ''' The proposed residual block from [4]'''
    y0 = Convolution2D(nfeats, ksize, ksize, border_mode='same')(x)
    y = y0
    for i in range(nskipped):
        y = BatchNormalization(mode=0, axis=1)(y)
        y = Activation('relu')(y)
        y = Convolution2D(nfeats, ksize, ksize, border_mode='same')(y)
    return merge([y0, y], mode='sum')

def getwhere(x):
    ''' Calculate the "where" mask that contains switches indicating which
    index contained the max value when MaxPool2D was applied.  Using the
    gradient of the sum is a nice trick to keep everything high level.'''
    y_prepool, y_postpool = x
    return K.gradients(K.sum(y_postpool), y_prepool)

# input image dimensions
img_rows, img_cols = 4096, 3328

# the data, shuffled and split between train and test sets
(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 4095
X_test /= 4095
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# The size of the kernel used for the MaxPooling2D
pool_size = 2
# The total number of feature maps at each layer
nfeats = [8, 16, 32, 64, 128]
# The sizes of the pooling kernel at each layer
pool_sizes = np.array([1, 1, 1, 1, 1]) * pool_size
# The convolution kernel size
ksize = 3
# Number of epochs to train for
nb_epoch = 5
# Batch size during training
batch_size = 128

if pool_size == 2:
    # if using a 5 layer net of pool_size = 2
    X_train = np.pad(X_train, [[0, 0], [0, 0], [2, 2], [2, 2]],
                     mode='constant')
    X_test = np.pad(X_test, [[0, 0], [0, 0], [2, 2], [2, 2]], mode='constant')
    nlayers = 5
elif pool_size == 3:
    # if using a 3 layer net of pool_size = 3
    X_train = X_train[:, :, :-1, :-1]
    X_test = X_test[:, :, :-1, :-1]
    nlayers = 3
else:
    import sys
    sys.exit("Script supports pool_size of 2 and 3.")

# Shape of input to train on (note that model is fully convolutional however)
input_shape = X_train.shape[1:]
# The final list of the size of axis=1 for all layers, including input
nfeats_all = [input_shape[0]] + nfeats

# First build the encoder, all the while keeping track of the "where" masks
img_input = Input(shape=input_shape)

# We push the "where" masks to the following list
wheres = [None] * nlayers
y = img_input
for i in range(nlayers):
    y_prepool = convresblock(y, nfeats=nfeats_all[i + 1], ksize=ksize)
    y = MaxPooling2D(pool_size=(pool_sizes[i], pool_sizes[i]))(y_prepool)
    wheres[i] = merge([y_prepool, y], mode=getwhere,
                      output_shape=lambda x: x[0])

# Now build the decoder, and use the stored "where" masks to place the features
for i in range(nlayers):
    ind = nlayers - 1 - i
    y = UpSampling2D(size=(pool_sizes[ind], pool_sizes[ind]))(y)
    y = merge([y, wheres[ind]], mode='mul')
    y = convresblock(y, nfeats=nfeats_all[ind], ksize=ksize)

# Use hard_simgoid to clip range of reconstruction
y = Activation('hard_sigmoid')(y)

# Define the model and it's mean square error loss, and compile it with Adam
model = Model(img_input, y)
model.compile('adam', 'mse')

# Fit the model
model.fit(X_train, X_train, validation_data=(X_test, X_test),
          batch_size=batch_size, nb_epoch=nb_epoch)

# Plot
X_recon = model.predict(X_test[:25])
X_plot = np.concatenate((X_test[:25], X_recon), axis=1)
X_plot = X_plot.reshape((5, 10, input_shape[-2], input_shape[-1]))
X_plot = np.vstack([np.hstack(x) for x in X_plot])
plt.figure()
plt.axis('off')
plt.title('Test Samples: Originals/Reconstructions')
plt.imshow(X_plot, interpolation='none', cmap='gray')
plt.savefig('reconstructions.png')
