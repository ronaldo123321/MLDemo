# -*- coding: utf-8 -*-

import cv2
import sys
from PIL import Image


def CatchUsbVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    #告诉OpenCV使用人脸识别分类器   haarcascade_frontalface_alt_tree.xml(检测最严格的分类器)
    #haarcascade_frontalface_alt2          haarcascade_frontalface_alt2.xml
    classfier = cv2.CascadeClassifier("F:\python3.7\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml");

    #识别出人脸后画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0

    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        #将当前帧转换为灰度图像(转一维降低计算强度)
        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #人脸检测，1.2和2分别为图像缩放比例和需要检测的有效点数
        #scaleFactor：图像缩放比例，可以理解为同一个物体与相机距离不同，其大小亦不同，必须将其缩放到一定大小才方便识别，该参数指定每次缩放的比例
        #minNeighbors 对特征检测点周边多少有效点同时检测，可避免因选取的特征检测点太小而导致遗漏
        #minSize：特征检测点的最小值
        faceRects = classfier.detectMultiScale(frame,scaleFactor=1.2,minNeighbors=5,minSize=(32,32))
        if len(faceRects) > 0:   #大于0则检测到人脸
            for faceRect in faceRects: ##单独框出每一张人脸
                x,y,w,h = faceRect

                #将当前帧保存为图片
                img_name = '%s/%d.jpg' % (path_name, num)
                image = frame[y-10 : y+h+10, x-10 : x+w+10]
                #完成实际图片的保存
                cv2.imwrite(img_name,image)
                num += 1
                if num > (catch_pic_num):  # 如果超过指定最大保存数量退出循环
                    break
                #画出矩形框
                cv2.rectangle(frame,(x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                # 显示当前捕捉到了多少人脸图片
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

        #超过指定最大保存数量则结束程序
        if num > catch_pic_num:
            break
        # 显示图像并等待10毫秒按键输入，输入‘q’退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


#python的str是utf-8编码，imshow的窗口标题是gbk编码。
def zh_cn(string):
    return string.encode("gbk").decode(errors="ignore")

#camera_id，这个就是USB摄像头的索引号，一般是0，如果0不行可以试试1、2等
if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    # else:
    #     CatchUsbVideo("截取视频流", int(sys.argv[0]))
        CatchUsbVideo(zh_cn("抓取视频流"), 0 ,1000,"F:\pyworkspace\data\yzt")