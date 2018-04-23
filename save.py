"""
Save predicted result to excel file.
"""

import utils

utils.write_to_excel(folder_holds_results='/home/panziqi/project/fashion_ai/version_softmax/result/test/',
                     path_to_train_excel_file='/home/panziqi/project/fashion_ai/annotations/train/train.xlsx',
                     path_to_orig_test_excel_file='/home/panziqi/project/fashion_ai/test.xlsx',
                     save_path='/home/panziqi/project/fashion_ai/version_softmax/result/test/test.csv')