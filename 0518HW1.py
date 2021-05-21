import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

# 使用 OpenCV 讀取圖檔
img_bgr = cv2.imread('Bird feeding 3 low contrast.tif')
# 因為opencv跟matplot顯示方法不同 先轉換BGR圖片為RGB圖片
img_rgb = img_bgr[:, :, ::-1]

plt.imshow(img_rgb)
plt.show()

plt.hist(img_rgb.ravel(), 256, [0, 256])  # 將三維陣列轉換成一維陣列以符合直方圖邊界為0-256 計算資料出現的次數
plt.title('origin')
plt.show()

new = (np.arctan((img_rgb - 128.0) / 32.0))
plt.imshow(new)
plt.show()

plt.hist(new.ravel(), 256, [0, 256])  # 將三維陣列轉換成一維陣列以符合直方圖邊界為0-256
plt.title('output')
plt.show()

print("Figure of s= T(r)")

x = np.arange(0, 255, 0.1)

y = (np.arctan((x - 128.0)/32.0))

plt.title("Figure of s=T(r)")
plt.xlabel("r")
plt.ylabel("s")
plt.plot(x, y)
plt.show()

df = pd.DataFrame({"r" : x, "s" : y})
df.to_csv("result.csv",index=False) #最後利用pandas輸出成excel
