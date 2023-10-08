# Author : E5H4N
# Author URI : https://github.com/35h4n
import os

import numpy as np
import cv2


def train_and_test(train, test, v=1):
    var_distance = []

    for i in range(train.shape[0]):
        dx = train[i, :-1]
        dy = train[i, -1]
        d = distance_calculation(test, dx)
        var_distance.append([d, dy])

    dv = sorted(var_distance, key=lambda k: k[0])[:v]
    l = np.array(dv)[:, -1]
    o = np.unique(l, return_counts=True)
    index = np.argmax(o[1])
    return o[0][index]


def distance_calculation(x, y):
    return np.sqrt(((x - y) ** 2).sum())


capture = cv2.VideoCapture(0)
fc = cv2.CascadeClassifier("e5h4n.xml")
dset = "./e5h4n_detection/"

fd = []
l = []
cid = 0
n = {}

for fx in os.listdir(dset):
    if fx.endswith('.npy'):
        n[cid] = fx[:-4]
        item = np.load(dset + fx)
        fd.append(item)

        t = cid * np.ones((item.shape[0],))
        cid += 1
        l.append(t)

fd_set = np.concatenate(fd, axis=0)
fl = np.concatenate(l, axis=0).reshape((-1, 1))
print(fl.shape)
print(fd_set.shape)

trset = np.concatenate((fd_set, fl), axis=1)
print(trset.shape)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    rt, frm = capture.read()
    if not rt:
        continue
    g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(g, 1.3, 5)

    for i in f:
        x, y, w, h = i
        oset = 5
        f_sec = frm[y - oset:y + h + oset, x - oset:x + w + oset]
        f_sec = cv2.resize(f_sec, (100, 100))
        out = train_and_test(trset, f_sec.flatten())

        cv2.putText(frm, n[int(out)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frm, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.imshow("Faces", frm)
    if cv2.waitKey(1) == ord('s'):
        break

capture.release()
cv2.destroyAllWindows()
