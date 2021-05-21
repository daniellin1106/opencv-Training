
import matplotlib.pyplot as plt
import cv2

img_input = cv2.imread('Bird 3 blurred.tif')#遮罩只留下BGR陣列中的B數值
b_image = img_input
for x in range(800): #圖片寬800，高1200
  for y in range(1200):
    b_image[x][y][1] = 0
    b_image[x][y][2] = 0
img_b = cv2.cvtColor(b_image, cv2.COLOR_BGR2GRAY)

img_input = cv2.imread('Bird 3 blurred.tif')#遮罩只留下BGR陣列中的G數值
g_image = img_input
for x in range(800):
  for y in range(1200):
    g_image[x][y][0] = 0
    g_image[x][y][2] = 0
img_g = cv2.cvtColor(g_image, cv2.COLOR_BGR2GRAY)

img_input = cv2.imread('Bird 3 blurred.tif')#遮罩只留下BGR陣列中的R數值
r_image = img_input
for x in range(800):
  for y in range(1200):
    r_image[x][y][0] = 0
    r_image[x][y][1] = 0
img_r = cv2.cvtColor(r_image, cv2.COLOR_BGR2GRAY)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(100, 10), nrows=1, ncols=3)#設定輸出圖像的寬高，列數與行數
ax1.imshow(img_b, cmap='gray') #以灰色漸層表示
ax1.set_title('B_Image')
ax1.set_xticks(range(0, 1200, 200))
ax1.set_yticks(range(0, 800, 200))

ax2.imshow(img_g, cmap='gray')
ax2.set_title('G_Image')
ax2.set_xticks(range(0, 1200, 200))
ax2.set_yticks(range(0, 800, 200))

ax3.imshow(img_r, cmap='gray')
ax3.set_title('R_Image')
ax3.set_xticks(range(0, 1200, 200))
ax3.set_yticks(range(0, 800, 200))

plt.show()

img_rgb = cv2.imread('Bird 3 blurred.tif')
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
h_image = img_hsv
for x in range(800):
  for y in range(1200):
    h_image[x][y][1]=0
    h_image[x][y][2]=0
img_h = cv2.cvtColor(h_image, cv2.COLOR_BGR2GRAY)

img_rgb = cv2.imread('Bird 3 blurred.tif')
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
s_image = img_hsv
for x in range(800):
  for y in range(1200):
    s_image[x][y][0]=0
    s_image[x][y][2]=0
img_s = cv2.cvtColor(s_image, cv2.COLOR_BGR2GRAY)

img_rgb = cv2.imread('Bird 3 blurred.tif')
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
i_image = img_hsv
for x in range(800):
  for y in range(1200):
    i_image[x][y][0]=0
    i_image[x][y][1]=0
img_i = cv2.cvtColor(i_image, cv2.COLOR_BGR2GRAY)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(100, 10), nrows=1, ncols=3)
ax1.imshow(img_h, cmap='gray')
ax1.set_title('H_Image')
ax1.set_xticks(range(0, 1200, 200))
ax1.set_yticks(range(0, 800, 200))

ax2.imshow(img_s, cmap='gray')
ax2.set_title('S_Image')
ax2.set_xticks(range(0, 1200, 200))
ax2.set_yticks(range(0, 800, 200))

ax3.imshow(img_i, cmap='gray')
ax3.set_title('I_Image')
ax3.set_xticks(range(0, 1200, 200))
ax3.set_yticks(range(0, 800, 200))

plt.show()

import numpy as np
from skimage.metrics import structural_similarity #新版本已經不支援from skimage.measure import compare_ssim，記得要改成左式

img = cv2.imread('Bird 3 blurred.tif')
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #使用kernal 2D filter方法得到
img_rgb_sharpened = cv2.filter2D(img, -1, kernel)

img_rgb = cv2.imread('Bird 3 blurred.tif')
img_hsi = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)#轉換成hsi顯示
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img_hsi_sharpened = cv2.filter2D(img_hsi, -1, kernel)
img_hsi_sharpened_rgb = cv2.cvtColor(img_hsi_sharpened, cv2.COLOR_HSV2BGR)

grayA = cv2.cvtColor(img_rgb_sharpened, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(img_hsi_sharpened_rgb, cv2.COLOR_BGR2GRAY)
(score, diff) = structural_similarity(grayA, grayB, full=True)#因為返回的資料型態會是0~1所以要*255換成色碼代號
difference = (diff * 255).astype("uint8")

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(100, 10), nrows=1, ncols=3)
ax1.imshow(img_rgb_sharpened, cmap = 'gray')
ax1.set_title('RGB-based sharpened images')
ax1.set_xticks(range(0, 1200, 200))
ax1.set_yticks(range(0, 800, 200))

ax2.imshow(img_hsi_sharpened_rgb, cmap = 'gray')
ax2.set_title('HSI-based sharpened images')
ax2.set_xticks(range(0, 1200, 200))
ax2.set_yticks(range(0, 800, 200))

ax3.imshow(difference, cmap = 'gray')
ax3.set_title('images difference')
ax3.set_xticks(range(0, 1200, 200))
ax3.set_yticks(range(0, 800, 200))

plt.show()

