import os
import sys
from PIL import Image   # 使用第三方包Pillow来进行图像处理
import numpy as np
import tensorflow.contrib.keras as k
import scipy.spatial.distance as distance   # 使用第三方包scipy来进行向量余弦相似度判断
import pandas as pd


# 这是一个自定义函数，用于把图片按比例缩放到最大宽度为maxWidth、最大高度为maxHeight
def resizeImage(inputImage, maxWidth, maxHeight):
    originalWidth, originalHeight = inputImage.size

    f1 = 1.0 * maxWidth / originalWidth
    f2 = 1.0 * maxHeight / originalHeight
    factor = min([f1, f2])

    width = int(originalWidth * factor)
    height = int(originalHeight * factor)
    return inputImage.resize((width, height), Image.ANTIALIAS)


ifRestartT = False
roundCount = 20
learnRate = 0.01
trainDir = "./data/"   # 用于指定训练数据所在目录
trainResultPath = "./imageClassify/"     # 用于指定训练过程保存目录
optimizerT = "RMSProp"  # 用于指定优化器
lossT = "categorical_crossentropy"  # 用于指定误差函数
#predictFile = None  # 用于指定需预测的新图像文件，如果为None则表示不预测
predictFile = "./predict/xx.jpg"

#  读取meta.txt文件内容：分类
#metaData = pd.read_csv(trainDir + "meta.txt", header=None).as_matrix()
metaData = pd.read_csv(trainDir + "meta.txt", header=None).iloc[:,:].values

# maxTypes表示种类个数
maxTypes = len(metaData)
print("maxTypes: %d" % maxTypes)

argt = sys.argv[1:]
print("argt: %s" % argt)

for v in argt:
    if v == "-restart":
        print("我打印了吗？")
        ifRestartT = True
    if v.startswith("-round="):
        roundCount = int(v[len("-round="):])
    if v.startswith("-learnrate="):
        learnRate = float(v[len("-learnrate="):])
    if v.startswith("-dir="):   # 用于指定训练数据所在目录（不使用默认目录时才需要设置）
        trainDir = v[len("-dir="):]
    if v.startswith("-optimizer="):
        optimizerT = v[len("-optimizer="):]
    if v.startswith("-loss="):
        lossT = v[len("-loss="):]
    if v.startswith("-predict="):
        predictFile = v[len("-predict="):]

print("predict file: %s" % predictFile)

xData = []
yTrainData = []
fnData = []
predictAry = []

listt = os.listdir(trainDir)    # 获取变量trainDir指定的目录下所有的文件

lent = len(listt)

# 循环处理训练目录下的图片文件，统一将分辨率转换为256*256，
# 再把图片处理成3通道（RGB）的数据放入xData，然后从文件名中提取目标值放入yTrainData
# 文件名放入fnData
for i in range(lent):
    v = listt[i]
    if v.endswith(".jpg"):  # 只处理.jpg为扩展名的文件
        print("processing %s ..." % v)
        img = Image.open(trainDir + v)
        w, h = img.size

        img1 = resizeImage(img, 256, 256)

        img2 = Image.new("RGB", (256, 256), color="white")

        w1, h1 = img1.size

        img2 = Image.new("RGB", (256, 256), color="white")

        img2.paste(img1, box=(int((256 - w1) / 2), int((256 - h1) / 2)))

        xData.append(np.matrix(list(img2.getdata())))

        tmpv = np.full([maxTypes], fill_value=0)
        tmpv[int(v.split(sep="-")[0]) - 1] = 1
        yTrainData.append(tmpv)

        fnData.append(trainDir + v)

rowCount = len(xData)
print("rowCount: %d" % rowCount)

# 转换xData、yTrainData、fnData为合适的形态
xData = np.array(xData)
xData = np.reshape(xData, (-1, 256, 256, 3))

yTrainData = np.array(yTrainData)

fnData = np.array(fnData)

# 使用Keras来建立模型、训练和预测
if (ifRestartT is False) and os.path.exists(trainResultPath + ".h5"):
    # 载入保存的模型和可变参数
    print("模型已经存在！！！！！！！！...")
    print("Loading...")
    model = k.models.load_model(trainResultPath + ".h5")
    model.load_weights(trainResultPath + "wb.h5")
else:
    # 新建模型
    model = k.models.Sequential()

    # 使用4个卷积核、每个卷积核大小为3*3的卷积层
    model.add(k.layers.Conv2D(filters=4, kernel_size=(3, 3), input_shape=(256, 256, 3), data_format="channels_last", activation="relu"))

    model.add(k.layers.Conv2D(filters=3, kernel_size=(3, 3), data_format="channels_last", activation="relu"))

    # 使用2个卷积核、每个卷积核大小为2*2的卷积层
    model.add(k.layers.Conv2D(filters=2, kernel_size=(2, 2), data_format="channels_last", activation="selu"))

    model.add(k.layers.Flatten())
    # 此处的256没有改
    model.add(k.layers.Dense(256, activation='tanh'))

    model.add(k.layers.Dense(64, activation='sigmoid'))

    # 按分类数进行softmax分类
    model.add(k.layers.Dense(maxTypes, activation='softmax'))

    model.compile(loss=lossT, optimizer=optimizerT, metrics=['accuracy'])


if predictFile is not None:
    # 先对已有训练数据执行一遍预测，以便后面做图片相似度比对
    print("preparing ...")
    predictAry = model.predict(xData)

    print("processing %s ..." % predictFile)
    img = Image.open(predictFile)

    #  下面是对新输入图片进行预测
    img1 = resizeImage(img, 256, 256)

    w1, h1 = img1.size

    img2 = Image.new("RGB", (256, 256), color="white")

    img2.paste(img1, box=(int((256 - w1) / 2), int((256 - h1) / 2)))

    xi = np.matrix(list(img2.getdata()))
    xi1 = np.array(xi)
    xin = np.reshape(xi1, (-1, 256, 256, 3))

    resultAry = model.predict(xin)
    print("x: %s, y: %s" % (xin, resultAry))

    # 找出预测结果中最大可能的概率及其对应的编号
    maxIdx = -1
    maxPercent = 0

    for i in range(maxTypes):
        if resultAry[0][i] > maxPercent:
            maxPercent = resultAry[0][i]
            maxIdx = i

    # 将新图片的预测结果与训练图片的预测结果逐一比对，找出相似度最高的
    minDistance = 200
    minIdx = -1
    minFile = ""

    for i in range(rowCount):
        dist = distance.cosine(resultAry[0], predictAry[i])     # 用余弦相似度来判断两张图片预测结果的相近程度
        if dist < minDistance:
            minDistance = dist
            minIdx = i
            minFile = fnData[i]

    print("推测表情：%s，推断正确概率：%10.6f%%，最相似文件：%s，相似度：%10.6f%%" % (metaData[maxIdx][1], maxPercent * 100, minFile.split("\\")[-1], (1 - minDistance) * 100))

    sys.exit(0)

model.fit(xData, yTrainData, epochs=roundCount, batch_size=lent, verbose=2)

print("saving...")
model.save(trainResultPath + ".h5")
model.save_weights(trainResultPath + "wb.h5")