import matplotlib.pyplot as plt
from tifffile import imsave
import numpy as np
from os import listdir
print("File: Reader")


class Reader:
	def __init__(self, file):
		self.file = file
		self.frames = []
		self.majorDict = {}




	def getFrames(self):
		listDirectories = listdir(self.file)[1:]
		print(listDirectories)

		for video in listDirectories:
			frames = listdir(self.file + video)[1:]
			TempList = []
			skipFactor = 1
			for i  in range(int((len(frames))/skipFactor)):
				singleFrame = frames[i*skipFactor]
				ff = plt.imread(self.file + video + "/" + singleFrame)
				TempList.append(ff)
			self.majorDict[video] = TempList
		return self.majorDict

	def save(self, array):
		imsave("BS/Test.tif", array)


# from PIL import Image
# im = Image.fromarray(np.ones((158, 238)) * 0.5)
# im.save("BS/Test.tif")


#
# if len(self.frames) > 0:
# 	return self.frames
# frames = []
# for element in self.list_Directories:
# 	single_frame = plt.imread(self.file + element)
# 	frames.append(np.array(single_frame))
# self.frames = frames
# return frames
