"""
Used for referring.
"""

import sys
import utils
import cpm

# 1.configuration
category = sys.argv[1]
gpu_id = int(sys.argv[2])
batch_size = int(sys.argv[3])

norm_image_size = (368, 368)
belief_map_size = (46, 46)
keypoints_count = len(utils.keypoints_order[category])

# 2.load data for referring
test_data = utils.dataLoader(category=category,
                             path_to_excel_file='/home/panziqi/project/fashion_ai/new_dataset/test.xlsx',
                             images_prefix='/home/panziqi/project/fashion_ai/new_dataset/',
                             norm_image_size=norm_image_size,
                             belief_map_size=belief_map_size)

# 3.refer
cpm = cpm.CPM(network_name='CPM_' + category, stage_count=stage,
              norm_image_size=norm_image_size, belief_map_size=belief_map_size, keypoints_count=keypoints_count)
cpm.predict(test_data=test_data,
            log_folder='/home/panziqi/project/fashion_ai/version_softmax/log/test/epoch30_aug/',
            params_folder='/home/panziqi/project/fashion_ai/version_softmax/params/epoch30_bkp/',
            folder_holds_rst_pics='/home/panziqi/project/fashion_ai/version_softmax/result/test/new_test/images/' + category + '/',
            path_to_save_file='/home/panziqi/project/fashion_ai/version_softmax/result/test/new_test/' + category + '.result',
            batch_size=batch_size, gpu_id=gpu_id)
