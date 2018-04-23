# CPM-Clothes-Keypoints-Detection
Convolutional Pose Machine implemented for clothes key points detection.
## 1 依赖环境和库
Ubuntu 16.04, Python3.5 with  
* numpy (1.14.2)
* mxnet-cu80 
* matplotlib (2.1.2)
* imageio (2.3.0)
* scikit-image (0.13.1)
* xlrd (1.1.0)
* xlwt (1.3.0)
## 2 训练步骤说明
### 2.1 数据预处理
utils.py脚本中的dataLoader类用于加载原始图像数据并做预处理。该类的实例化参数有5个：
* category: 数据对应的衣服种类。枚举为'skirt', 'outwear', 'blouse', 'dress', 'trousers'.
* path_to_excel_file: 用于训练的train.xlsx文件绝对路径
* images_prefix: 存放训练图片Images的文件夹绝对路径
* norm_image_size: 网络输入归一化图像大小。本网络模型使用(368, 368).
* belief_map_size: 网络预测的热度图大小。本网络模型使用(46, 46)   

该类的实例化对象作为训练过程中的图像数据和标签数据来源。训练时每当取出一个批次数据之时首先读出原始图像，进行旋转、缩放等数据增强操作，并对真实的关键点做对应的处理；之后根据关键点生成真实热度图作为网络输出的训练标签。
### 2.2 网络训练过程
#### 2.2.1 初始化网络模型

## 3测试步骤说明
