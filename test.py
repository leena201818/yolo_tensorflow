import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class   #输出向量7*7*30分为三段：类向量，有对象概率，对象位置信息
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())                    #初始化变量

        print('Restoring weights from: ' + self.weights_file)               #必须要给定模型文件
        self.saver = tf.train.Saver()                                       #用于恢复模型文件
        self.saver.restore(self.sess, self.weights_file)

    #绘制检测目标框，result:list of bbox(class,x,y,w,h,pclass)
    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

    #探测结果，并将模型输出解释成list of bbox(class,x,y,w,h,pclass)形式
    def detect(self, img):
        #数据预处理
        img_h, img_w, _ = img.shape                                         #原始图片尺寸，需要resize到模型尺寸
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32) #颜色空间转换
        inputs = (inputs / 255.0) * 2.0 - 1.0                               #模型要求每个元素在0-1之间
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))#模型输入(batch=1,448,448,2)

        #执行一次模型推断，获取输出结果
        result = self.detect_from_cvmat(inputs)[0]                           #0表示batch=1只有一个样例

        #模型输出的尺度都是基于（448,448,3）的，需要转换成原始图片尺寸上的(x,y,w,h)
        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    #模型推断，从输出（None,1470）解析检测目标结果
    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):                    #(batch,1470)
            results.append(self.interpret_output(net_output[i]))

        return results

    #模型输出解释：主要理解！如何从(None,1470)中解析出探测到的目标
    def interpret_output(self, output):
        '''
        1、预测值：x,y,sqrt(w),sqrt(h)尺度都是基于cell_size=7的，即基于7*7大小的特征图片。
            模型预测出的(x,y)坐标是相对于所在grid的坐标，尺度基于7*7
            模型预测出来的sqrt(w),sqrt(h)，而非w,h.
        2、模型预测出的每个cell，有两个Pobj，一套calss_probs
        3、首先确定每个cell中两个bbox较大的Pobj，消去较小的（设置mask）
        4、对各cell的Pobj进行降序排列，按照0.2的概率进行截取
        5、基于Pobj排序，采用最大值抑制方式，消除重复bbox
        6、剩下的bbox就是想要的检测目标结果
        目标：从预测出来的boxes,class_probs,得到过滤后的boxes_filtered,prob_filtered和class_num_filtered
        '''
        probs = np.zeros((self.cell_size, self.cell_size,                       #用来存放各个bbox目标所属class概率[7,7,2,20]
                          self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(                                               #预测出的各个cell目标所属class概率[7,7,20]
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(                                                    #预测出的各个bbox有目标概率[7,7,2]
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(                                                     #预测出的各个bbox(x,y,sqrt(w),sqrt(h)),相对坐标，cell_size尺寸基准
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)
        offset = np.transpose(                                                  #[7,7,2]各个bbox的坐标偏置
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0))

        #bbox的各个尺度都按照7*7特征图坐标系偏移、尺度进行归一化
        boxes[:, :, :, 0] += offset                                             #预测出的bbox位置x,y校正成相对特征图坐标系（7,7）
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size         #预测出的bbox的x,y尺度校正成相对于cell_size归一化
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])                      #将预测出的sqrt(w),sqrt(h)变为w,h，方便计算iou

        boxes *= self.image_size                                                #将预测出的(x,y,w,h)尺度变为相对于模型输入大小（448,448）

        #probs[7,7,2,20],第i个bbox中第j个class的概率 =第i个bbox的Pobj*预测出的对应cell的第j个class概率。最终bbox类别概率=Pboj*Pclass
        #该形状便于mask
        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])                      #[7,7,20]*[7,7,2],前两个::表明对应cell
        #先通过probs[7,7,2,20]找到预测值最大的那几个目标，记住该位置，再找到对应的bbox和classid和prob
        #filter_mat_boxes代表[7,7,2,20]中选中目标的索引！
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')      #[7,7,2,20],仅仅提取超过阈值的bbox
        filter_mat_boxes = np.nonzero(filter_mat_probs)                         #返回选中对象的张量索引，以list形式返回，list[0]代表第一个维度
        boxes_filtered = boxes[filter_mat_boxes[0],                             #切片，返回选中对象对应的bbox(None,4)，boxes_filtered是选中bbox，其数量=bbox数量
                               filter_mat_boxes[1], filter_mat_boxes[2]]        #filter_mat_boxes[0]是[7,7,2,20]第一维的坐标,因为只写了3个维度，因此得到了最后维度的[None,4]
        probs_filtered = probs[filter_mat_probs]                                #切片，返回选中对象的对象概率(None,)，因为filter_mat_probs取全维度，所以直接索引到prob

        #np.argmax(filter_mat_probs, axis=3)针对[7,7,2,20]，找到第3维度最大的索引（Pobj->classid），输出[7,7,2]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[             #用filter_mat_boxes定位，获取[None,]classid
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]                    #probs_filtered是概率值，和boxes_filtered，classes_num_filtered位置对应
        #按照对象概率的顺序，对应相应的bbox和概率，以及classid
        boxes_filtered = boxes_filtered[argsort]                                #倒序排列选中的bbox[7,7,2,4]
        probs_filtered = probs_filtered[argsort]                                #倒序排列选中的bbox对应类概率[7,7,2,20]
        classes_num_filtered = classes_num_filtered[argsort]                    #倒序排列选中的bbox对应的类ID

        #boxes_filtered已经按照对象概率倒序排列。非最大值抑制，从大往小排列，后者如果和前者iou超过阈值，则剔除
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')       #剔除，获得最终结果，获取过滤后的位置索引
        boxes_filtered = boxes_filtered[filter_iou]                     #以位置索引得到最终结果
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        #返回检测对象[[classname,x,y,w,h,pobj]],这里的pobj=预测出的bbox中有对象概率× 20个分类的argmax概率
        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    #iou = inter / (area1+area2-inter)
    def iou(self, box1, box2):
        # left-right交集的宽：min(2矩形框右侧边沿)-max(2矩形框左侧边沿)
        lr = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        # top-bottom交集的高：min(2矩形框下册边沿)-max(2矩形框上侧边沿)
        tb = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    #从摄像机捕获视频 cap = cv2.VideoCapture(-1)
    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()                         #读取结果

        while ret:
            ret, frame = cap.read()                 #读取结果和图片帧
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)                       #wait>10表明等待10ms，如果0表明无限等待

            ret, frame = cap.read()

    #检测图片中目标，并绘制bbox
    #imname图片路径
    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    detector = Detector(yolo, weight_file)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    imname = 'test/person.jpg'
    detector.image_detector(imname)


if __name__ == '__main__':
    main()
