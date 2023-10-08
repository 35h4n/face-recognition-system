# Author : E5H4N
# Author URI : https://github.com/35h4n

import numpy as np
import cv2

capture = cv2.VideoCapture(0)
fc = cv2.CascadeClassifier("e5h4n.xml")

while True:
    rt, frm = capture.read()
    g_frm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    if not rt:
        continue
    f = fc.detectMultiScale(g_frm, 1.3, 5)
    if len(f) == 0:
        continue
    # k = 1
    # f = sorted(f, key=lambda x: x[2] * x[3], reverse=True)
    # skip += 1
    for i in f[:1]:
        x, y, w, h = i
        oset = 5
        f_sec = frm[y - oset:y + h + oset, x - oset:x + w + oset]
        f_sel = cv2.resize(f_sec, (100, 100))

        # if skip % 10 == 0:
        #     fd.append(f_sel)
        #     print(len(fd))
        # cv2.imshow(str(k), f_sel)
        # k += 1

    cv2.imshow("Face Recognition System : E5H4N", frm)
    # cv2.rectangle(frm, (x, y), (x + w, y + h), (255, 255, 255), 2)

    press = cv2.waitKey(1)
    if press == ord('s'):
        break
capture.release()
cv2.destroyAllWindows()
