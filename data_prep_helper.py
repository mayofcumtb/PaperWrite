#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import random
import tempfile
import shutil

mayan_debug = 1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from caffe_utils import *

'''
@brief: 
    extract labels from rendered images
@input: 
    xxx/03790512_13a245da7b0567509c6d15186da929c5_a035_e009_t-01_d004.png
@output: 
    (35,9,359)
'''
def path2label(path):
    parts = os.path.basename(path).split('_')
    azimuth = int(parts[2][1:]) % 360
    elevation = int(parts[3][1:]) % 360
    tilt = int(parts[4][1:]) % 360
    return (azimuth, elevation, tilt)
    
def outspath2label(path):
    '''

    :param path: input_labels+=bookshelf_16_a007_e023_t359_d002px_216.00_py_499.00_bbwidth_280.00_bbheight_485.00.jpg===
                              =====0======1==2=====3===4=====5======6====7===8=======9=======10======11======12=========
    :return:
    '''
    parts = os.path.basename(path).split('_')
    class_name = str(parts[0])
    cad_index = str(parts[1])
    azimuth = int(parts[2][1:]) 
    elevation = int(parts[3][1:])
    tilt = -int(parts[4][1:])
    distance = float(parts[5][1:-2])
    px = float(parts[6])
    py = float(parts[8])
    bbox_width = float(parts[10])
    bbox_height = float(parts[12][:-4])

    return (class_name, cad_index, azimuth, elevation, tilt, distance, px, py, bbox_width, bbox_height)


'''
@brief:
    get rendered image filenames and annotations, save to specified files.
@input:
    shape_synset - like '02958343' for car
    [train,test]_image_label_file - output file list filenames
    train_ratio - ratio of training images vs. all images
@output:
    save "<image_filepath> <class_idx> <azimuth> <elevation> <tilt>" to files.
'''
def get_one_category_image_label_file(shape_synset, train_image_label_file, test_image_label_file, train_ratio = 0.9):
    if mayan_debug:
        train_ratio = 1
    class_idx = g_shape_synsets.index(shape_synset) 

    image_folder = os.path.join(g_syn_images_bkg_overlaid_folder, shape_synset)
    all_md5s = os.listdir(image_folder)
    train_test_split = int(len(all_md5s)*train_ratio)
    train_md5s = all_md5s[0:train_test_split]
    test_md5s = all_md5s[train_test_split:]

    for md5s_list, image_label_file in [(train_md5s, train_image_label_file), (test_md5s, test_image_label_file)]:
        image_filenames = []
        for k,md5 in enumerate(md5s_list):
            if k%(1+len(md5s_list)/20)==0:
                print('shape: %s clsidx: %d, %d/%d: %s' % (shape_synset, class_idx, k,len(md5s_list),md5))
            shape_folder = os.path.join(image_folder, md5)
            shape_images = [os.path.join(shape_folder, x) for x in os.listdir(shape_folder)]
            image_filenames += shape_images
        image_filename_label_pairs = [(fpath,path2label(fpath)) for fpath in image_filenames]
        random.shuffle(image_filename_label_pairs)

        fout = open(image_label_file, 'w')
        for filename_label in image_filename_label_pairs:
            label = filename_label[1]
            fout.write('%s %d %d %d %d\n' % (filename_label[0], class_idx, label[0], label[1], label[2]));
        fout.close()
no_bkg = 0
def ours_get_one_category_image_label_file(shape_synset, train_image_label_file, test_image_label_file, train_ratio = 0.9):
    class_idx = g_shape_synsets.index(shape_synset)
    #image_source_file = "/data/zairan.wang/YanMA/RenderForCNN/data_ours/syn_images_cropped_bkg_overlaid_real/"
    
    image_folder = os.path.join(g_syn_images_bkg_overlaid_folder, shape_synset)
    all_md5s = os.listdir(image_folder)
    train_test_split = int(len(all_md5s)*train_ratio)
    train_md5s = all_md5s[0:train_test_split]
    test_md5s = all_md5s[train_test_split:]
    
#############path control##########################################################################
    all_file_paths = "/data/zairan.wang/YanMA/RenderForCNN/data_ours/VOC_format/SUN_bkg/all_info_cropped_bkg.txt"
    images_file = "/data/zairan.wang/YanMA/RenderForCNN/data_ours/VOC_format/SUN_bkg/Images_uniform_bkg_cropped/"
###########################################mayan########################################
    if os.path.exists(all_file_paths):
      f_all = open(all_file_paths, 'a')
    else:
      f_all = open(all_file_paths, 'w')
    for md5s_list, image_label_file in [(train_md5s, train_image_label_file), (test_md5s, test_image_label_file)]:
        image_filenames = []
        for k,md5 in enumerate(md5s_list):
            if k%(1+len(md5s_list)/20)==0:
                print('shape: %s clsidx: %d, %d/%d: %s' % (shape_synset, class_idx, k,len(md5s_list),md5))
            shape_folder = os.path.join(image_folder, md5)
            shape_images = [os.path.join(shape_folder, x) for x in os.listdir(shape_folder)]
            image_filenames += shape_images
        image_filename_label_pairs = [(fpath,outspath2label(fpath)) for fpath in image_filenames]
        random.shuffle(image_filename_label_pairs)

        #fout = open(image_label_file, 'w')
        for filename_label in image_filename_label_pairs:
            label = filename_label[1]
            shutil.copy(filename_label[0], images_file)
            f_all.write('%s %s %s %d %d %d %d %f %f %f %f\n' % (filename_label[0],  label[0], label[1], label[2], label[3], label[4], label[5], label[6], label[7], label[8], label[9]));
            #fout.write('%s %s %s %d %d %d %d %f %f %f %f\n' % (filename_label[0],  label[0], label[1], label[2], label[3], label[4], label[5], label[6], label[7], label[8], label[9]));
        #fout.close()
    f_all.close()


'''
@brief:
    combine lines from input files and save the shuffled version to output file.
@input:
    input_file_list - a list of input file names
    output_file - output filename
'''
def combine_files(input_file_list, output_file, shuffle=1):
    all_lines = []
    for filelist in input_file_list:
        lines = [x.rstrip() for x in open(filelist,'r')]
        all_lines += lines

    if shuffle: random.shuffle(all_lines)

    fout = open(output_file,'w')
    for line in all_lines:
        fout.write('%s\n' % (line))
    fout.close()

'''
@brief:
    convert 360 view degree to view estimation label
    e.g. for bicycle with class_idx 1, label will be 360~719
'''
def view2label(degree, class_index):
  return int(degree)%360 + class_index*360


'''
@brief:
    generate LMDB from files containing image filenames and labels
@input:
    image_label_file - each line is <image_filepath> <class_idx> <azimuth> <elelvation> <tilt>
    output_lmdb: LMDB pathname-prefix like xxx/xxxx_lmdb
    image_resize_dim (D): resize image to DxD square
@output:
    write TWO LMDB corresponding to images and labels, 
    i.e. xxx/xxxx_lmdb_label (each item is class_idx, azimuth, elevation, tilt) and xxx/xxxx_lmdb_image
'''
def generate_image_view_lmdb(image_label_file, output_lmdb):
    lines = [line.rstrip() for line in open(image_label_file,'r')]

    tmp_label_fout = tempfile.NamedTemporaryFile(dir=g_syn_images_lmdb_folder, delete=False)
    for line in lines:
        ll = line.split(' ')
        class_idx, azimuth, elevation, tilt = [int(x) for x in ll[1:]]
        tmp_label_fout.write('%d %d %d %d\n' % (class_idx, view2label(azimuth, class_idx), view2label(elevation, class_idx), view2label(tilt, class_idx)))
    tmp_label_fout.close()
    print("Tmp label file generated: %s" % tmp_label_fout.name)
    
    if not os.path.exists(output_lmdb+'_label'):
        write_vector_lmdb(tmp_label_fout.name, output_lmdb+'_label')
    print "Label DB done ..."
    if not os.path.exists(output_lmdb+'_image'):
        write_image_lmdb(image_label_file, output_lmdb+'_image')  
    print "Image DB done ..."
    
    # clean up
    os.system('rm %s' % (tmp_label_fout.name))
