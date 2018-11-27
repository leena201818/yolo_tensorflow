import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim

'''
    YOLO目标探测网络，one-stage
'''
class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL                            #每个cell探测的目标个数
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.num_class + self.boxes_per_cell * 5)                      #一维向量，S*S*(B*5+20),7*7*(5*2+20) = 1470
        self.scale = 1.0 * self.image_size / self.cell_size                 #从输入图片到输出特征图的尺寸缩放倍数 448/7=64
        self.boundary1 = self.cell_size * self.cell_size * self.num_class   #模型预测输出向量中分为三段：预测目标类型向量区，预测对象概率，预测对象box位置
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE                                #cell中有对象的损失权重 1
        self.noobject_scale = cfg.NOOBJECT_SCALE                            #cell中无对象的损失权重 1
        self.class_scale = cfg.CLASS_SCALE                                  #cell中对象分类损失权重 2
        self.coord_scale = cfg.COORD_SCALE                                  #cell中对象位置损失权重 5

        self.learning_rate = cfg.LEARNING_RATE                              #学习率
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA                                              #leak_relu激励函数的alpha

        #每个bbox的偏置,7*7*30拉平后每个bbox的位置;
        # self.labels(None,cell_size,cell_size,25)是ground true，因此lebels[:,:,:,i]只有25维。
        # ！重点理解：对于一个cell，预测2个目标框，预测的(x,y)是相对于所在cell的，它的绝对坐标需要加上所在cell的坐标
        # 这里的offset指的是预测的(x,y,w,h)所在cell在grid中的位置坐标offset，shape=[7,7,2],offset[:,:,0]=[0,1,2,3,4,5,6]

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        #模型输入张量(None,448,448,3)
        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')

        #模型输出张量（None,1470）,7*7*(5*2+20)
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])         #模型输出
            self.loss_layer(self.logits, self.labels)                               #在graph中添加损失张量节点
            self.total_loss = tf.losses.get_total_loss()                            #返回全部损失张量
            tf.summary.scalar('total_loss', self.total_loss)

    #定义网络结构，输入images,输出1470维向量
    #注意：1470的采用如下分布：7*7*20,7*7*2,7*7*2*4,分别表示每个cell中目标类别向量，cell中两个box的objectness概率,cell中两个box的位置
    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),     #在height,width维度上前后各pad 3个0
                    name='pad_1')
                net = slim.conv2d(
                    net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),    #分别在(batch，height，width，channels)维度上pad 0
                    name='pad_27')
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')  #（batch,height,width,channels）->(batch,channels,height,width)
                net = slim.flatten(net, scope='flat_32')                #在将(None,7,7,1024)的特征图拉平之前，将同一个channel的特征图聚合在一起！相当于1024张特征图
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                net = slim.fully_connected(
                    net, num_outputs, activation_fn=None, scope='fc_36')    #输出张量（None,7*7*30)
        return net

    #计算一个批量样本的输出特征grid各cell内box的IOU,矩阵运算，可以将[batch,cell_size,cell_size,boxex_per_cell,4]当成[...,4]一个4维向量来对待
    #返回[batch,cell,cell,boxes_per_cell]，表示一批数据中，每个网格的2个候选框和ground true 目标框之间的iou
    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point，计算相交矩形框的左上和右下角坐标
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)             #如果左上在右下的右侧，没有交集！
            inter_square = intersection[..., 0] * intersection[..., 1]  #注意：此处发生变化

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]       #注意：此处形状发生变化shape:[batch,cell_size,cell_size,boxes_per_cell]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)          #iou在0-1.0之间

    #损失层，输入参数：
    #predicts:（None,1470）,其中1470=7*7*(5*2+20)，每个cell两个同类备选框
    #predicts:[类向量，预测box是对象的概率，对象位置],一维向量，分为三个部分，强制模型输出，位置预测的是（x,y,sqrt(w),sqrt(h)）,x,y是相对于grid的坐标
    #labels:(None,7,7,25),真值(objectness,x,y,h,w,class_vector)，每个cell只能有一个对象
    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            #预测值：x,y,sqrt(w),sqrt(h)尺度都是基于cell_size=7的，即基于7*7大小的特征图片。
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])          #预测目标的类型向量（batch,7,7,20）
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])     #预测box中objectness概率(0,1) (batch,7,7,2)
            predict_boxes = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])  #预测box的（x,y,2,sqrt(w),sqrt(h)） (batch,7,7,2,4)

            #实际值：ground true的x,y,w,h都是基于image_size，即基于448*448大小的原始图片
            #为了和预测值计算损失，都需要进行归一化（/image_size）
            response = tf.reshape(
                labels[..., 0],                                                             #ground true的objectness (batch,7,7,1),值0或者1
                [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(
                labels[..., 1:5],                                                           #ground true的（x,y,w,h）(batch,7,7,1,4)
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(                                                                #这里将groundtrue的bbox位置/image_size,可以为了对应模型预测输出，(x,y,w,h)都除各自图片大小，归一对齐
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size                 #ground true的box只有一份，但是预测出的是2个box，因此需要复制
            classes = labels[..., 5:]                                                       #ground true的class vector (batch,...)

            offset = tf.reshape(                                                            #self.offset.shape=(7,7,2),2表明2个box,每个[0,1,2,...,6]
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])                   #一个样例的偏置(1,7,7,2)

            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])                            #一批中所有样例(batch,7h,7w,2),grid的行索引
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))                                #行列号互换,(batch,7w,7h,2),grid的列索引
            predict_boxes_tran = tf.stack(                                                  #将list of tensor 堆砌成R+1维tensor
                [(predict_boxes[..., 0] + offset) / self.cell_size,                         #predict_boxes[..., 0]x
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,                    #predict_boxes[..., 1]y
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)                                #模型预测的是（x,y,sqrt(h),sqrt(w)),计算iou时需要平方

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)                    #计算一批数据的iou,[batch,cell,cell,boxex_per_cell]

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]         #I张量，表明每个预测box有对象的概率
            object_mask = tf.reduce_max(iou_predict_truth, axis=3, keep_dims=True)          #沿着boxex_per_cell轴，计算最大值，保留该维度，但该维度长度reduce变为1，[BATCH_SIZE, CELL_SIZE, CELL_SIZE, 1]
            object_mask = tf.cast(                                                          #计算公式中的Iobj，是对象的概率
                (iou_predict_truth >= object_mask), tf.float32) * response                  #tf.cast变成0,1，乘以ground_ture的objectness，和真值之间配对。这里iou_predict_truth >= object_mask时，后者进行了broadcasting在axis=3
                                                                                            #response里面只有ground true的0,1,response在axis=3也进行了broadcasting

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]                  #box中预测框无对象的概率,Pnoobj = 1-Pobj
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack(                                                          #ground true转换成相对image_size的尺度和相对于grid的坐标，因为预测输出的就是相对于grid的坐标
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss 对象类型误差：均方差误差
            class_delta = response * (predict_classes - classes)                            #这里不是采用object_mask，而是response，表明针对cell，而非bbox
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),                      #针对cell_size,cell_size，boxes_per_cell求和,针对axis每个轴，结果会减少1维，本次操作共减少3维
                name='class_loss') * self.class_scale

            # object_loss bbox预测是对象的误差
            object_delta = object_mask * (predict_scales - iou_predict_truth)               #网络直接预测出的predict_scales概率，和实际objectness概率的误差
            object_loss = tf.reduce_mean(                                                   #而实际objectness概率是通过预测bbox和真实bbox计算iou表示的，预测objectness和bbox相互制约
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss box非对象的预测误差
            noobject_delta = noobject_mask * predict_scales                                 #predict_scales设计成是将box预测成对象的概率，那么将box预测成非对象的误差就是predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss    bbox位置误差
            coord_mask = tf.expand_dims(object_mask, axis=4)                                #增加1维[BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL,1]
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),                   #在[CELL_SIZE, CELL_SIZE, BOXES_PER_CELL,1]上求和
                name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])                      #boxes_delta位置偏差
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


#定义操作图节点op（input）输出张量
def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
