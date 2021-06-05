import cv2 as cv
import numpy as np
import xlsxwriter
import math
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

count = 0
x = np.zeros((10000, 1))
kernel_ed = np.ones((5,5),np.uint8)


def rescale(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


def get_length(orig, c, count):
    # Lấy minRect
    box = cv.minAreaRect(c)

    # Lấy tọa độ các đỉnh của MinRect
    box = cv.boxPoints(box)

    # Sắp xếp các điểm theo trình tự
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box

    # Vẽ min rect
    cv.drawContours(orig, [box.astype("int")], 0, (0,0,255), 2)

    # Tính chiều rộng
    x[count] = (dist.euclidean(tl, bl) + dist.euclidean(tl, tr))/2

    # In ra giá trị
    cv.putText(orig, "{:.2f}".format(x[count,0]), (0, 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    count = count + 1

    return orig, count


def process(frame, count):
    # Tiền xử lý
    frame2 = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    #frame2 = cv.medianBlur(frame2, 25)
    frame2 = cv.medianBlur(frame2, 5)
    #frame2 = cv.GaussianBlur(frame2, (9, 9), 0)
    ret, frame2 = cv.threshold(frame2, 250, 255, cv.THRESH_BINARY)
    cv.imshow("Threshold",frame2)
    frame2 = cv.erode(frame2, kernel_ed)
    frame2 = cv.dilate(frame2, kernel_ed)
    cv.imshow("Close",frame2)

    # Tìm đường bao
    edged = cv.Canny(frame2, 50, 100)
    cv.imshow("Edged", edged)
    cnts, hierarchy = cv.findContours(edged, 1, 2)

    # Tính chiều dài hcn bao và vẽ lên hình
    frame, count = get_length(frame, cnts[0], count)
    # for c in cnts:
    #     if cv.contourArea(c) > 200:
    #         frame, count = get_length(frame, c, count)
    #         break

    return frame, count

cap = cv.VideoCapture('Photo/video.mp4')


# check = 1
while(cap.isOpened()):
    isTrue, frame = cap.read()
    if isTrue:
        frame = rescale(frame)
        img = frame[310:430,570:720]
        frame2, count = process(img, count)
    
    # Để lưu video
    # if check == 1:
    #     out = cv.VideoWriter('Vid.mp4', cv.VideoWriter_fourcc(*'X264'), 20.0, (int(frame2.shape[1]), int(frame2.shape[0])))
    #     check = 0
    # out.write(frame2)

        cv.imshow("Source", frame)
        cv.imshow("Processed", frame2)
        
    if cv.waitKey(20) & 0xff == ord('q'): break

plt.plot(range(1, count), x[0:count-1])
plt.show()
# Lưu trong excel
# workbook = xlsxwriter.Workbook('Data2.xlsx')
# worksheet = workbook.add_worksheet()
# for i in range(count):
#     worksheet.write(i, 0, x[i])
# workbook.close()

cap.release()
# out.release()
cv.destroyAllWindows()