import os

#
# path and dataset parameter
#

# DATA_PATH = '/home/mika/PycharmProjects/yolo_tensorflow/data'
DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')

# WEIGHTS_FILE = None
WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 448

CELL_SIZE = 7

BOXES_PER_CELL = 2      #每个cell探测的对象个数

ALPHA = 0.1             #leaky_relu激励函数的alpha f(x) = alpla*x whenx<=0

DISP_CONSOLE = True

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = '0'

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000             #指数衰减学习率的延迟步，每30000步进行一次学习率衰减

DECAY_RATE = 0.1                #衰减系数decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

STAIRCASE = True

# BATCH_SIZE = 45
BATCH_SIZE = 45

#MAX_ITER = 15000
MAX_ITER = 150

# SUMMARY_ITER = 10             #每SUMMARY_ITER在日志中记录摘要，每SUMMARY_ITER×10在console中计算损失，打印摘要
SUMMARY_ITER = 3

#SAVE_ITER = 1000
SAVE_ITER = 100                 #每当SAVE_ITER迭代，保存一次模型参数


#
# test parameter
#

THRESHOLD = 0.2                 #探测对象概率阈值，超过该阈值，进入最大值抑制候选.预测对类别概率=Pobj*Pclass

IOU_THRESHOLD = 0.5             #非最大值抑制的阈值，后面的bbox和前者重叠iou待遇该阈值，则剔除
