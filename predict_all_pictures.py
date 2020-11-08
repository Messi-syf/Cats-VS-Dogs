
import shutil
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *
import glob
from keras.models import load_model
import os


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

for n in range(1,12493,9):
    for i in range(n,n+9):
        shutil.move('test/%d.jpg'%(i), 'test4')



    # 忽略硬件加速的警告信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    file_path = 'test4/'
    f_names = glob.glob(file_path + '*.jpg')

    img = []
    # 把图片读取出来放到列表中
    for i in range(len(f_names)):
        # load the image
        imag = image.load_img(f_names[i], target_size=(224, 224))
        # convert to array
        x = image.img_to_array(imag)
        # reshape into a single sample with 3 channels
        x = x.reshape(1, 224, 224, 3)
        # center pixel data
        x = x.astype('float32')
        x = x - [123.68, 116.779, 103.939]
        img.append(x)




    # 把图片数组联合在一起
    x = np.concatenate([x for x in img])

    model = load_model('best_model.h5')
    y = model.predict(x)
    print('Predicted:', y)
    with open('sample_submission.csv','a',encoding='utf-8') as f:
        for i in y:
            f.write(str(int(i[0]))+'\n')



    del_file('test4/')

