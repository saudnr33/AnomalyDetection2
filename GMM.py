import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from Reader import Reader
from sklearn.cluster import KMeans



from numpy import savetxt


file = Reader("UCSDped1/Train/")
data_dictionary = file.getFrames()
keys = data_dictionary.keys()

majorArray = []

k = 5

z = 0
for key in keys:
    for i in range( int(len( data_dictionary[key] )/k)):
        Temp = data_dictionary[key][i:i+k]

        Temp = np.array(Temp).reshape(1, -1)
        majorArray.append(Temp[0])
        print(Temp)
majorArray = np.array(majorArray)
# savetxt('data.csv', majorArray, delimiter=',')

print(np.shape(majorArray))
print("here")
clf = KMeans(n_clusters=10)
clf.fit(majorArray)
print("done")



file = Reader("UCSDped1/TestModified/")
data_dictionary = file.getFrames()
keys = data_dictionary.keys()

majorArray = []

z = 0
for key in keys:
    for i in range( int(len( data_dictionary[key] )/k)):
        Temp = data_dictionary[key][i:i+k]
        Temp = np.array(Temp).reshape(1, -1)
        print(np.shape(Temp))
        print("key = ", key, "; i = ", i , ";  " , clf.score(Temp), clf.predict(Temp))

# file = Reader("UCSDped1/TestModified/")
# data_dictionary = file.getFrames()
# keys = data_dictionary.keys()
# print()
