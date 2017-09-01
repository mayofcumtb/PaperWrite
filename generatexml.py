#-*- coding:utf-8 -*-
import xml.dom
import xml.dom.minidom
import os
# from PIL import Image
import cv2
import shutil

tv_debug = 0
# xml 默认配置信息，如有需要可以修??_TXT_PATH= 'all_info.txt'


_INDENT= ''*4
_NEW_LINE= '\n'
_FOLDER_NODE= 'VOC2007'
_ROOT_NODE= 'annotation'
_DATABASE_NAME= 'SunBKG_SYN_IMAGE'
_ANNOTATION= 'PASCAL VOC2007'
_AUTHOR= 'MAYan'
_SEGMENTED= '0'
_DIFFICULT= '0'
_TRUNCATED= '0'
#_POSE= 'Unspecified'

# 保存路径Images_uniform_no_bkg_cropped/

_IMAGE_PATH= 'Images_uniform_no_bkg_no_crop'
_IMAGE_COPY_PATH= 'JPEGImages_uniform_no_bkg_no_crop'
_ANNOTATION_SAVE_PATH= 'Annotations_uniform_no_bkg_no_crop'
_TEXTFILE_SAVE_PATH = "TextFile_uniform_no_bkg_no_crop"
mayan_debug = 1
mayan_rotate_azimuth_debug = 0
def createElementNode(doc,tag, attr):  # 创建一个元素节??    
    element_node = doc.createElement(tag)

    # 创建一个文本节节点
    text_node = doc.createTextNode(attr)

    # 将文本节点作为元素节点的子节??    
    element_node.appendChild(text_node)

    return element_node

    # 封装添加一个子节点的过??
def createChildNode(doc,tag, attr,parent_node):



    child_node = createElementNode(doc, tag, attr)

    parent_node.appendChild(child_node)

# object节点比较特殊

def createObjectNode(doc,attrs):

    object_node = doc.createElement('object')

    createChildNode(doc, 'name', attrs['classification'],
                    object_node)
    pose_node = doc.createElement('pose')

    createChildNode(doc, 'azimuth', attrs['azimuth'],
                    pose_node)

    createChildNode(doc, 'elevation', attrs['elevation'],
                    pose_node)

    createChildNode(doc, 'distance', attrs['distance'],
                    pose_node)

    createChildNode(doc, 'theta', attrs['theta'],
                    pose_node)
   # createChildNode(doc, 'pose',
  #                  _POSE, object_node)
    object_node.appendChild(pose_node)

    createChildNode(doc, 'truncated',
                    _TRUNCATED, object_node)

    createChildNode(doc, 'difficult',
                    _DIFFICULT, object_node)
   
    createChildNode(doc, 'occluded',
                    '0', object_node)
    createChildNode(doc, 'cad_index',
                    attrs['cad_index'], object_node)
    bndbox_node = doc.createElement('bndbox')

    createChildNode(doc, 'xmin', attrs['xmin'],
                    bndbox_node)

    createChildNode(doc, 'ymin', attrs['ymin'],
                    bndbox_node)

    createChildNode(doc, 'xmax', attrs['xmax'],
                    bndbox_node)

    createChildNode(doc, 'ymax', attrs['ymax'],
                    bndbox_node)

    object_node.appendChild(bndbox_node)

    return object_node

# 将documentElement写入XML文件??
def writeXMLFile(doc,filename):

    tmpfile =open('tmp.xml','w')

    doc.writexml(tmpfile, addindent=''*4,newl = '\n',encoding = 'utf-8')

    tmpfile.close()

    # 删除第一行默认添加的标记

    fin =open('tmp.xml')

    fout =open(filename, 'w')

    lines = fin.readlines()

    for line in lines[1:]:

        if line.split():

         fout.writelines(line)

        # new_lines = ''.join(lines[1:])

        # fout.write(new_lines)

    fin.close()

    fout.close()

def getFileList(path):

    fileList = []
    files = os.listdir(path)
    for f in files:
        if (os.path.isfile(path + '/' + f)):
            fileList.append(f)
    # print len(fileList)
    return fileList
def outspath2label(path):
    '''

    :param path: input_labels+=bookshelf_16_a007_e023_t359_d002px_216.00_py_499.00_bbwidth_280.00_bbheight_485.00.jpg===
                              =====0======1==2=====3===4=====5======6====7===8=======9=======10======11======12=========
    :return:
    '''
    if mayan_rotate_azimuth_debug:
        rotate_deg = {
           'bed':[180,90,180,90,180,90,90,90,90,90,90,90,90,90,180,180,90,90,90,90],
           'bench':[90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90],
           'bookshelf':[90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90],
           'cabinet':[90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90],
           'chair':[90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90],
           'sofa':[90,90,90,90,90,90,90,90,90,0,90,90,90,90,90,90,90,90,90,90],
           'table':[90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90],
           'tvmonitor':[90,90,90,90,270,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90],
           'pillow':[90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90]
        }
	  
    parts = path.split('_')
    class_name = str(parts[0])
    cad_index = str(parts[1])
    if mayan_rotate_azimuth_debug:
        temp_list = rotate_deg[class_name]
        temp_value = temp_list[int(cad_index)-1]
    azimuth = int(parts[2][1:]) % 360
    if mayan_rotate_azimuth_debug:
        azimuth = (azimuth - temp_value) % 360
    if tv_debug:
        if class_name == "tvmonitor":
            azimuth = azimuth +180
            azimuth = azimuth % 360
    elevation = int(parts[3][1:])
    tilt = int(parts[4][1:])
    distance = float(parts[5][1:-2])
    #TODO need to be modified
    px = float(parts[6])
    py = float(parts[8])
    bbox_width = float(parts[10])
    bbox_height = float(parts[12][:-4])
#    x_min = px - bbox_width
#    x_max = px + bbox_width
#    y_min = py - bbox_height
#    y_max = py + bbox_height
#    px = x_min+(bbox_width/2)
#    py = y_min+(bbox_height/2)
    x_min = px - bbox_width/2
    y_min = py - bbox_height/2
    x_max = px + bbox_width/2
    y_max = py + bbox_height/2
    #print azimuth
    #import pdb; pdb.set_trace(); 
    
    if azimuth>=180:
        azimuth = azimuth-360
      #  print azimuth
    result = {
        'classification': class_name,
        'cad_index': cad_index,
        'azimuth': '%d'%azimuth,
        'elevation': '%d'%elevation,
        'theta': '%d'%tilt,
        'distance': '%.2f'%distance,
        'px': '%.2f'%px,
        'py': '%.2f'%py,
        'xmin': '%.2f'%x_min,
        'ymin': '%.2f'%y_min,
        'xmax': '%.2f'%x_max,
        'ymax': '%.2f'%y_max
    }

    return result

if __name__ == "__main__":
    if not os.path.exists(_TEXTFILE_SAVE_PATH):
        os.mkdir(_TEXTFILE_SAVE_PATH)
    f = open(os.path.join(_TEXTFILE_SAVE_PATH, 'all_name.txt'),'w')
    if not os.path.exists(_ANNOTATION_SAVE_PATH):
        os.mkdir(_ANNOTATION_SAVE_PATH)

    if not os.path.exists(_IMAGE_COPY_PATH):
        os.mkdir(_IMAGE_COPY_PATH)
    
    file_list = getFileList(_IMAGE_PATH)
    
    for index in range(len(file_list)):
        if index % 500 == 0:
            print "%d in %d" % (index, len(file_list))
        temp_file_path = file_list[index]
        save_name = "s%06d" % (index)
        f.write(save_name + '\n')
        pos = temp_file_path.rfind('.')
        textName = temp_file_path[:pos]
        xml_file_name = os.path.join(_ANNOTATION_SAVE_PATH, (save_name + '.xml'))
        
        img = cv2.imread(os.path.join(_IMAGE_PATH, temp_file_path))
        height, width, channel = img.shape
        # print os.path.join(_IMAGE_COPY_PATH,(save_name+'.jpg'))
        cv2.imwrite(os.path.join(_IMAGE_COPY_PATH, (save_name + '.jpg')), img)
        my_dom = xml.dom.getDOMImplementation()
        doc = my_dom.createDocument(None, _ROOT_NODE, None)
        # 根节??        
        root_node = doc.documentElement

        # folder节点
        

	# createChildNode(doc, 'folder',_FOLDER_NODE, root_node)

	# filename节点

        createChildNode(doc, 'filename', save_name+'.jpg',root_node)

        # size节点

        size_node = doc.createElement('size')

        createChildNode(doc, 'width', str(width), size_node)

        createChildNode(doc, 'height', str(height), size_node)

        createChildNode(doc, 'depth', str(channel), size_node)

        root_node.appendChild(size_node)

        attrs = outspath2label(temp_file_path)
        if index % 500 == 0:
            print attrs
        object_node = createObjectNode(doc, attrs)
        root_node.appendChild(object_node)
        writeXMLFile(doc, xml_file_name)
        if index % 50 == 0:
            try:
                os.path.exists(xml_file_name)
            except:
                print "python has unseen error in codes"
            else:
                print "all is well"
    f.close()