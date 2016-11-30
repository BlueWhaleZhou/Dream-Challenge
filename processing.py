import numpy as np
import pydicom as dcm
import cv2

# processing dcm data
'''
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
'''
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
'''
filename = path + file_list[0]
data_sample = dcm.read_file(filename).pixel_array
data_sample = data_sample.astype('float32')
data_matrix = data_sample * 255 / 4095
ret, data_matrix = cv2.threshold(data_matrix, 249.084, 255, cv2.THRESH_TOZERO_INV)
print (data_matrix.shape)
'''
shape = (2048000, 3328)
matrix = np.zeros(shape)
for i in range(500):
    name_temp = path + file_list[i]
    sample_temp = dcm.read_file(name_temp)
    data_temp = sample_temp.pixel_array.astype('float32')
    data_temp = data_temp * 255.0 / 4095.0
    ret, data_temp = cv2.threshold(data_temp, 249.084, 255, cv2.THRESH_TOZERO_INV)
    if data_temp.shape is (3328, 2560):
        data_temp_reshape = cv2.copyMakeBorder(data_temp, 0, 768, 0, 768, cv2.BORDER_CONSTANT, value=0)
        print (data_temp_reshape.shape)
        matrix[i * 4096: i * 4096 + 4096, :] = data_temp_reshape
    else:
        matrix[i * 4096: i * 4096 + 4096, :] = data_temp
    matrix /= 255.0
print (matrix.shape)
print (matrix)
np.savetxt('/home/qinghai/research/dream/dream.txt', matrix, delimiter=",")
