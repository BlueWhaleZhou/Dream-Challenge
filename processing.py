import numpy as np
import os
import pydicom
import glob

list = []

path = "/home/qinghai/research/dream/pilot_images/"
for filename in glob.glob(os.path.join(path, "*.dcm")):
    list.append(filename)
#print list
print (len(list))

for i in range(len(list)):
    file = pydicom.read_file(list[i])
    data = file.pixel_array
    print (data.shape)



