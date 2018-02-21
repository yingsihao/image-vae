import os

allfile=[]
cnt1 = 0

def getallfile(path):
  allfilelist=os.listdir(path)
  for file in allfilelist:
    filepath=os.path.join(path,file)
#判断是不是文件夹
    if os.path.isdir(filepath):
      getallfile(filepath)
    else:
      global cnt1
      cnt1 += 1
      allfile.append(filepath)
      os.system("mv " + filepath + " " + str(cnt1) + ".jpg")
  return allfile
 
if __name__ == '__main__':
  path="./"
  allfiles=getallfile(path)
 
  #for item in allfiles:
  #  print(item)
  print(len(allfiles))
