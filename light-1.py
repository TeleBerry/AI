import cv2
import numpy as np

# 匯入
img = cv2.imread('c:\\opencv_one\\pic\\light.jpg', -1)

# 先將照片轉灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 環形檢測
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    1,
    20,
    None,
    10,  # 高閥值
    6,  # 超過此閥值才會當一個圓
    6,  # min 半徑
    7  # max 半徑
)

# 變回彩色
if len(circles) > 0:
    out = img.copy()
    for x, y, r in circles[0]:
        cv2.circle(out, (x, y), int(r), (255, 0, 0), 2, cv2.LINE_AA)
        img = cv2.hconcat([img, out])

# 輸出
out = cv2.resize(out, (1000, 400), interpolation=cv2.INTER_AREA)
cv2.imshow('frame', out)

cv2.waitKey(0)
cv2.destroyAllWindows()
