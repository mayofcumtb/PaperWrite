import os
import shutil

classes = os.listdir('../test')
for class_name in classes:
  if not os.path.exists(class_name):
    os.mkdir(class_name)
  for name in classes:
      if name == class_name:
        if os.path.isdir(os.path.abspath(os.path.join('../test',class_name))):
          #print "file_exist"
        # import pdb;pdb.set_trace()
        #os.symlink(os.path.abspath(os.path.join('../test',class_name)),os.path.join(class_name,class_name))
          shutil.copytree(os.path.abspath(os.path.join('../test',class_name)),os.path.join(class_name,class_name))
      else:
        os.mkdir(os.path.join(class_name,name))
  