# CPM-Clothes-Keypoints-Detection
An MXNet implementation of Convolutional Pose Machine for clothes key points detection. 
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
* category: 数据对应的衣服种类。枚举为'skirt', 'outwear', 'blouse', 'dress', 'trousers'
* path_to_excel_file: 用于训练的train.xlsx文件绝对路径
* images_prefix: 存放训练图片Images的文件夹绝对路径
* norm_image_size: 网络输入归一化图像大小。本网络模型使用(368, 368)
* belief_map_size: 网络预测的热度图大小。本网络模型使用(46, 46)   

该类的实例化对象作为训练过程中的图像数据和标签数据来源。训练时每当取出一个批次数据之时首先读出原始图像，进行旋转、缩放等数据增强操作，并对真实的关键点做对应的处理；之后根据关键点生成真实热度图作为网络输出的训练标签。
### 2.2 网络训练过程
#### 2.2.1 初始化网络模型
网络模型使用4个阶段的Convolutional Pose Machine. 模型结构如model.pdf中所示意。cpm.py脚本中的CPM类用于网络模型的结构、前向传播过程、训练过程、预测过程等操作的封装实现。该类的实例化参数如下：
* network_name: 网络名称，用于网络参数文件的保存等。
* stage_count: 网络模型包含的阶段个数。
* norm_image_size: 网络输入归一化图像大小。本网络模型使用(368, 368)
* belief_map_size: 网络预测的热度图大小。本网络模型使用(46, 46)
* keypoints_count: 关键点个数。  
#### 2.2.2 网络训练过程
网络训练时调用CPM类实例对象的train方法。该方法的参数许下：
* train_data: 训练数据，其是utils.py脚本中的dataLoader类实例对象。
* log_folder: 存放训练日志的文件夹绝对路径。
* params_folder: 存放网络训练参数的文件夹绝对路径。
* epochs: 训练轮数。
* batch_size: 批次大小。
* ctx: 训练设备，其是list类实例对象，多个训练设备（如GPU）的集合。
* init_lr: 初始化学习率。本模型使用0.015.
* lr_step: 学习率变化所经过的训练轮数。本模型使用3.
* lr_factor: 每次学习率衰减时变化为原来的倍数。本模型使用0.7.   

网络训练结束后，会在params_folder中生成对应的网络参数训练文件。
#### 2.2.3 训练过程说明
网络使用MXNet自带的交叉熵损失函数SoftmaxCrossEntropyLoss作为损失函数。训练采用随机梯度下降(sgd)方法、动量为0.9. CPM的网络每个阶段会输出一个(46, 46)大小的热度图，将预测热度图与真实的热度图做交叉熵损失，并采用中继监督训练，将所有阶段的损失叠加到一起同时反向传播，以防止梯度消失的问题。
## 3测试步骤说明
