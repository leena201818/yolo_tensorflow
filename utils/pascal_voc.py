import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg

'''
    解析pascal_voc数据，转换成对象探测的格式，并生成data_generator
'''
class pascal_voc(object):
    def __init__(self, phase, rebuild=False):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE    #批大小
        self.image_size = cfg.IMAGE_SIZE    #模型输入图片大小
        self.cell_size = cfg.CELL_SIZE      #模型输出特征图的网格大小
        self.classes = cfg.CLASSES          #voc数据集对象类
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes)))) #{class_name:class_id}
        self.flipped = cfg.FLIPPED  #是否在训练集中添加翻转图片
        self.phase = phase      #train,test
        self.rebuild = rebuild  #是否将原始图片中的目标信息转换成模型输出格式（一张图片对应一个(cell_size,cell_size,25)张量）
        self.cursor = 0         #在一轮读取中，训练样本的游标
        self.epoch = 1          #数据generator读取时，记录训练轮次
        self.gt_labels = None   #图片及对象标注的索引，
        self.prepare()

    #循环读取训练数据 images(batch,height,width,channel),labels(batch,cell_size,cell_size,25)
    #每一个cell的:objectness+x,y,w,h+20个class共25维，这里的objectness表示该网格中是否有对象，ground true,只取0,1
    #注意：yolo的不足，每个cell只能有一个对象。如果输入图片翻转，对应的模型输出特征图也要翻转，包括对应的cell对象label也要翻转
    #label是ground true，因此只有25维。而对于一个cell，可能预测cfg.boxes_per_cell个目标框，预测和真值之间有offset
    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):      #如果超过数据，就重新读取，并增加轮次，最后一批，确保每次都能读取batch_size张图片
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    #读取图片，翻转，sacle尺寸，归一化，翻转输入图片
    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]                                   #水平翻转，倒叙排列cols
        return image

    #数据准备：附加水平翻转的训练输出特征图片。读取输出特征、目标标注、水平翻预处理(全部)、append,打散样本
    def prepare(self):
        gt_labels = self.load_labels()                                  #字典（图片path，label，flipped）
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :] #翻转第2列cell，label：(7,7,25)

                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:    #对objectness=1有对象的cell，计算翻转后的x，（y,w,h）不变
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]

            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)                                    #随机打散
        self.gt_labels = gt_labels
        return gt_labels

    #图片的目标标注信息gt_labels是个字典列表（list of dictionary）,dictionary is {imname:图片路径,flipped:True/False,label：（cell_size,cell_size,25）}
    #特别注意：label里面的x,y,w,h已经映射到了最终的格子，期间经过图片到resize的尺度变化以及经过卷积降采样的尺度变化。original imagesize->normal imagesize->(conv pool)cell_size*cell_size
    # 这里的目标根据中心点（x,y）确定放置在那个cell里面，有目标的cell obj=1
    def load_labels(self):
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    #返回一张图片的目标信息，转换成模型训练需要的结构（None,cell_size,cell_size,25）
    #原始图片尺寸->标准图片尺寸->特征图片尺寸，每个box的x,y,w,h都要经过以上变换，放置在相应的cell中
    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        index:图片的编号
        """

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')

            # 坐标从0开始，原始坐标转换成normal规范化模型输入图片坐标
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            #该box的目标类型
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]

            # 计算目标中心点所在的cell，这里从模型输入经过conv+pool降采样特征图尺寸变为7*7
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            #每个cell只能有一个目标，obj=0表明该cell还没有目标
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            #注意：这里的类id已经采用了one-hot编码！
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)

if __name__ == '__main__':
    p = pascal_voc('train', rebuild=False)

    gt_labels = p.load_labels()

    gt_labels = p.prepare()

    for i in range(3):
        images, labels = p.get()

    print('End')