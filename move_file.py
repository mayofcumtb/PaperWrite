#import shutil
import os

annotation_file_path = "/data/zairan.wang/YanMA/RenderForCNN/data_ours/VOC_format/SUN_bkg/Annotations_uniform_no_bkg_cropped"
image_file_path = "/data/zairan.wang/YanMA/RenderForCNN/data_ours/VOC_format/SUN_bkg/JPEGImages_uniform_no_bkg_cropped"
text_file_path = "/data/zairan.wang/YanMA/RenderForCNN/data_ours/VOC_format/SUN_bkg/TextFile_uniform_no_bkg_cropped"

#target_file_path = "/data/zairan.wang/YanMA/syn_datas/SUN_bkg/"
target_file_path = "/data/zairan.wang/YanMA/syn_datas/VOC_syn_uniform_cropped_no_bkg/"
if not os.path.exists(target_file_path):
    os.mkdir(target_file_path)
target_annotation_file_path = target_file_path+'Annotations'
target_image_file_path = target_file_path+'JPEGImages'
target_text_file_path = target_file_path+'TextFile'

#shutil.copy(annotation_file_path)
if os.path.exists(target_annotation_file_path):
    os.system('rm -rf %s' % (target_annotation_file_path))
if os.path.exists(target_image_file_path):
    os.system('rm -rf %s' % (target_image_file_path))
if os.path.exists(target_text_file_path):
    os.system('rm -rf %s' % (target_text_file_path))
#os.system('rm -rf %s %s %s'%(annotation_file_path, image_file_path, text_file_path))
os.system('cp -r %s %s %s %s'%(annotation_file_path, image_file_path, text_file_path, target_file_path))