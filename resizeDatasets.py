""" 
	script to resize the images in the datasets
	from (128*128*3) to (64*64*3)
"""

import os
import shutil
from PIL import Image
from pathlib import Path

# resizes image in filepath to width, maintains aspect ratio
def resizeImage(filepath, width=64):
	basewidth = width
	img = Image.open(filepath)
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((basewidth,hsize), Image.ANTIALIAS)
	img.save(filepath) 


def main():
	#create folder
	#Path("./resizedData").mkdir(parents=True, exist_ok=True)

	#copy all files to another folder
	srcFolder = "./Data"
	dstFolder = "./resizedData"
	try:
		shutil.copytree(srcFolder, dstFolder)
	except:
		print("Directory %s already exists. Cancelling copy\n", dstFolder)

	for root, dirs, files in os.walk(dstFolder):
		for f in files:
			#filepath
			filepath = root+"/"+f
			resizeImage(filepath, 64)






if __name__ == '__main__':
	main()