import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from ucsd_dataset import UCSDAnomalyDataset
from video_CAE import VideoAutoencoderLSTM
import torch.backends.cudnn as cudnn
import numpy as np
#matplotlib notebook
import matplotlib.pyplot as plt
from Labels import Labels


print("Testing Starts Now!")

model = torch.load("SavedModel")

test_ds = UCSDAnomalyDataset('UCSDped1/Test2')
test_dl = data.DataLoader(test_ds, batch_size=1, shuffle=False)





frames = []
errors = []
z = 0
print(len(test_dl))
k = 7
batch_idx = 0

for x in test_dl:
    print(batch_idx)
    if batch_idx == 191 * k:
        break
    # if z == 5:
    #     break
    y = model(x.cuda())
    mse = torch.norm(x.cpu().data.view(x.size(0),-1) - y.cpu().data.view(y.size(0),-1), dim=1)
    errors.append(mse)
    batch_idx += 1
errors = torch.cat(errors).numpy()
print(np.shape(errors))
errors = errors.reshape(-1, 191)
s = np.zeros((7,191))
for i in range(7):

    s[i,:] =  (errors[i,:] - np.min(errors[i,:]))/(np.max(errors[i,:]) - np.min(errors[i,:]))
# s[1,:] =  (errors[1,:] - np.min(errors[1,:]))/(np.max(errors[1,:]) - np.min(errors[1,:]))
# s[2,:] =  (errors[2,:] - np.min(errors[2,:]))/(np.max(errors[2,:]) - np.min(errors[2,:]))
# s[3,:] =  (errors[3,:] - np.min(errors[3,:]))/(np.max(errors[3,:]) - np.min(errors[3,:]))

th = np.linspace(0, 1, 150)
def labeler(a, t):
    out = []
    for x in a:
        if x > t:
            out.append(1)
        else:
            out.append(0)
    return out
AllLabels = Labels().labels(1)
Acc = []
TruePos = []
TrueNeg = []
MajAcc = np.zeros(len(th))
MajPos = np.zeros(len(th))
MajNeg = np.zeros(len(th))
recMaj = np.zeros(len(th))
for j in range(len(s)):
    Acc = []
    TruePos = []
    TrueNeg = []
    recall = []
    for t in th:
        out = labeler(s[j,:], t)
        label = AllLabels[j]
        correct_0 = 0
        correct_1 = 0
        n_0 = 0
        n_1 = 0
        e = 0
        for i in range(len(out)):
            if label[i + 5] == 1:
                n_1 +=1
                if out[i] == 1:
                    correct_1+=1
            else:
                n_0 +=1
                if out[i] == 0:
                    correct_0+=1
                else:
                    e +=1
        TruePos.append(correct_1/n_1)
        TrueNeg.append(correct_0/n_0)
        if correct_1+ e == 0:
            recall.append(1)
        else:

            recall.append(correct_1 / (correct_1+ e))
        Acc.append(    (correct_1 + correct_0 ) /(n_0 + n_1)   )

    if j == 0:
        for b in range(16):
            MajAcc += Acc
            MajNeg += TrueNeg
            MajPos += TruePos
            recMaj += recall
    MajAcc += Acc
    MajNeg += TrueNeg
    MajPos += TruePos
    recMaj += recall


index = np.argmax(MajAcc)
d = len(s) + 16
print(MajAcc[index]/d,MajPos[index]/d, MajNeg[index]/d )
ticks = []
for i in range(0, 190, 10):
    ticks.append(i)







xx = 1 - MajNeg/d
yy = MajPos/d
summ = 0
for i in range(len(xx) - 1):
    print((yy[i+1] ), (xx[i+1] - xx[i]))
    summ += (yy[i+1] )* (xx[i+1] - xx[i])


print(summ)


plt.figure(figsize=(3,6))
plt.xlim(0,190)
plt.xticks(ticks)
plt.plot(s[5,:])
plt.show()


plt.plot(s[1,:])
plt.show()



# plt.figure(figsize=(3,6))
# plt.xlim(0,190)
# plt.xticks(ticks)
# plt.plot(s[3,:])
# plt.show()

plt.plot(th, MajAcc/d, "k--")
plt.plot(th, MajAcc/d, "k.")
plt.show()

plt.plot(th, MajNeg/d)
plt.title("Correctly Detecting Normal Frame")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.show()

plt.plot(th, MajPos/d)
plt.title("Correctly Detecting Anomoly")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.show()

plt.plot(1 - MajNeg/d,  MajPos/d, "o")
plt.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), "-")
plt.title("ROC")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.show()
