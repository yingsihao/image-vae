#coding:utf-8
import os
allfile=[]
cnt=0

def getallfile(path):
  allfilelist=os.listdir(path)
  for file in allfilelist:
    filepath=os.path.join(path,file)
#判断是不是文件夹
    if os.path.isdir(filepath):
      getallfile(filepath)
    else:
      allfile.append(filepath)
      global cnt
      cnt += 1
      if cnt % 10 >= 1:
        os.system("cp \"" + filepath + "\" ../data/train/")
      else:
        os.system("cp \"" + filepath + "\" ../data/val/")
  return allfile
 
if __name__ == '__main__':
  path="./"
  allfiles=getallfile(path)
 
  #for item in allfiles:
  #  print(item)
  print(len(allfiles))