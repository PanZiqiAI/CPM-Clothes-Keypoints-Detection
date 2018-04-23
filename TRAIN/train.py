"""
main
"""

import sys
import utils
import cpm
import mxnet as mx

# 1.configuration
category = sys.argv[1]
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
gpu_s = int(sys.argv[4])
gpu_e = int(sys.argv[5])

norm_image_size = (368, 368)
belief_map_size = (46, 46)
keypoints_count = len(utils.keypoints_order[category])


# 2.load data for training
train_data = utils.dataLoader(category=category,
                              path_to_excel_file='/home/panziqi/project/fashion_ai/annotations/train/train.xlsx',
                              images_prefix='/home/public/FashionAI/keypoint/season1/train/',
                              norm_image_size=norm_image_size,
                              belief_map_size=belief_map_size)

# 3.train
cpm = cpm.CPM(network_name='CPM_' + category,
              stage_count=stage,
              norm_image_size=norm_image_size, belief_map_size=belief_map_size,
              keypoints_count=keypoints_count)
cpm.train(train_data=train_data,
          log_folder='/home/panziqi/project/fashion_ai/version_softmax/log/train/',
          params_folder='/home/panziqi/project/fashion_ai/version_softmax/params/',
          epochs=epochs, batch_size=batch_size,
          init_lr=0.015, lr_step=3, lr_factor=0.7,
          ctx=[mx.gpu(gpu_id) for gpu_id in range(gpu_s, gpu_e)])
