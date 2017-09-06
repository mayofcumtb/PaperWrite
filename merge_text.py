import os

voc_2007_text = '/disk_array/sdg/YanMA/datasets/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
voc_2012_text = '/disk_array/sdg/YanMA/datasets/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
target_file = 'val.txt'
fw = open(target_file,'w')
fr = open(voc_2007_text)
context = fr.readlines()
for element in context:
    fw.write(element)
fr.close()
fr2 = open(voc_2012_text)
context = fr2.readlines()
for element in context:
    fw.write(element)
fw.close()
