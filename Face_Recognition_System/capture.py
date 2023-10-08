# Author : E5H4N
# Author URI : https://github.com/35h4n


import cv2

capture = cv2.VideoCapture(0)
while True:
    rt, frm = capture.read()
    if not rt:
        continue

    cv2.imshow("Face Recognition System : E5H4N", frm)
    press = cv2.waitKey(1)

    if press == ord('s'):  # press 's' to exit
        break

capture.release()
cv2.destroyAllWindows()
