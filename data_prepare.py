
import os
import shutil

import numpy as np

from math import ceil

from PIL import Image

from collections import Counter
import matplotlib.pyplot as plt


from keras.preprocessing.image import *




os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ## 数据探索
# 在做计算机视觉的任务时，第一步很重要的事情是看看要做的是什么样的数据集，是非常干净的？还是存在各种遮挡的？
# 猫和狗是大是小？图片清晰度一般怎么样？会不会数据集中有标注错误的数据，比例是多少？散点分布情况？




imagelist = os.listdir('train/')

total_width = 0
total_height = 0
avg_width = 0
avg_height = 0
count = 0
min_width = 10000
min_height = 10000
max_width = 0
max_height = 0
min_product = 100000001
min_product_width = 0
min_product_height = 0

for x in imagelist:
    image_name = 'train/' + x
    image = Image.open(image_name)
    width = image.size[0]
    height = image.size[1]
    total_width += width
    total_height += height
    if min_width > width:
        min_width = width
    if min_height > height:
        min_height = height
    if max_width < width:
        max_width = width
    if max_height < height:
        max_height = height
    if min_product > width * height:
        min_product = width * height
        min_product_width = width
        min_product_height = height
    count += 1
print(count)
avg_width = total_width / count
avg_height = total_height / count
print("avg_width={}\navg_height={}\nThe total number of image is {}".format(avg_width, avg_height, count))
print("The min width is {}\nThe max width is {}".format(min_width, max_width))
print("The min height is {}\nThe max height is {}".format(min_height, max_height))
print("The min image size is {}*{}".format(min_product_width, min_product_height))

# train 训练集包含了 25000 张猫狗的图片，平均宽=404px，平均高=360px，最小的宽=42px，最大宽=1050px，最小高=32px，最大高=768px；
# 可以发现很多分辨率低图片，我们需要清理掉这些的图片。

# #### 绘制训练集中所有图片的大小散点图分布情况：

train_image_list = os.listdir('train/')
height_array = []
width_array = []
for name in train_image_list[1:]:
    image = load_img('train/' + name)
    x = img_to_array(image)
    height_array.append(x.shape[0])
    width_array.append(x.shape[1])

x = np.array(width_array)
y = np.array(height_array)
area = np.pi * (15 * 0.05) ** 2

plt.scatter(x, y, s=area, alpha=0.5, marker='x')
plt.show()

# #### 找出训练集中的所有长 or 宽 < 70px 的图片，分析其清晰度


train_image_list = os.listdir('train/')
bad_pictures = []
for name in train_image_list[1:]:
    image = load_img('train/' + name)
    x = img_to_array(image)
    if x.shape[0] < 70 or x.shape[1] < 70:
        bad_pictures.append(name)
print(bad_pictures)



# #### 去除离群点
# 异常对象被称作离群点。异常检测也称偏差检测和例外挖掘。孤立点是一个明显偏离与其他数据点的对象,它就像是由一个完全不同的机制生成的数据点一样。
# 离群点检测是数据挖掘中重要的一部分，它的任务是发现与大部分其他对象显著不同的对象。


plt.style.use('seaborn-white')

train_image_list = os.listdir('train/')
ratio_list = []

for name in train_image_list[1:]:
    image = Image.open('train/' + name)
    x = image.histogram(mask=None)
    count = Counter(x)
    ratio = float(len(count)) / len(x)
    ratio_list.append(ratio)

# np.percentile获取百分位数
q99, q01 = np.percentile(a=ratio_list, q=[99, 1])
print(q99, q01)

# In[20]:


# 将异常图片输出
plt.style.use('seaborn-white')

outlier_images = []
train_image_list = os.listdir('train/')
for name in train_image_list[:]:
    image = Image.open('train/' + name)
    x = image.histogram(mask=None)
    count = Counter(x)
    ratio = float(len(count)) / len(x)
    if ratio < q01:
        outlier_images.append(name)


print(outlier_images)



# 定义创建目标路径方法
def mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)



# 创建异常图片集文件夹
mkdir('outlier')


# 挑出所有低分率的图片
outlier_list = ['cat.10107.jpg', 'cat.10277.jpg', 'cat.10392.jpg', 'cat.10893.jpg', 'cat.11091.jpg', 'cat.11184.jpg',
                'cat.2433.jpg', 'cat.2939.jpg', 'cat.3216.jpg', 'cat.4821.jpg', 'cat.4833.jpg', 'cat.5534.jpg',
                'cat.6402.jpg', 'cat.6699.jpg', 'cat.7703.jpg', 'cat.7968.jpg', 'cat.8138.jpg', 'cat.8456.jpg',
                'cat.8470.jpg', 'cat.8504.jpg', 'cat.9171.jpg', 'dog.10190.jpg', 'dog.10654.jpg', 'dog.10747.jpg',
                'dog.11248.jpg', 'dog.11465.jpg', 'dog.11686.jpg', 'dog.1174.jpg', 'dog.12331.jpg', 'dog.1308.jpg',
                'dog.1381.jpg', 'dog.1895.jpg', 'dog.3074.jpg', 'dog.4367.jpg', 'dog.4507.jpg', 'dog.5604.jpg',
                'dog.630.jpg', 'dog.6685.jpg', 'dog.7772.jpg', 'dog.8450.jpg', 'dog.8736.jpg', 'dog.9188.jpg',
                'dog.9246.jpg', 'dog.9517.jpg', 'dog.9705.jpg']



# 将这些异常图片从训练集中删除
plt.figure(figsize=(12, 20))
outlier_image_size = len(outlier_list)
for i in range(0, outlier_image_size):
    plt.subplot(ceil(outlier_image_size / 6), 6, i + 1)
    img = load_img('train/' + outlier_list[i])
    x = img_to_array(img)
    plt.title(outlier_list[i])
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(img, interpolation="nearest")
    shutil.move('train/' + outlier_list[i], 'outlier/' + outlier_list[i])