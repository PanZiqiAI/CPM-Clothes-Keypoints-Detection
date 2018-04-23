"""
Convolutional pose machine model
"""

import os
import pickle
import utils
import logging
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon import trainer
from mxnet.gluon import loss
from mxnet.gluon.utils import split_and_load


class CPM_stage(nn.Block):
    """
    Stage block for CPM
    """
    def __init__(self, network_name, keypoints_count, stage, block_public_feature=None, **kwargs):
        """
        :param network_name:
        :param stage:
        :param keypoints_count:
        :param kwargs:
        """
        super(CPM_stage, self).__init__(**kwargs)
        # member variables
        self._network_name = network_name
        self._stage = stage
        self._keypoints_count = keypoints_count
        # used for average pooling
        self._block_averagePool = None
        # used for belief map
        self._block_beliefMap = None
        # used for dependent feature extracting
        self._block_feature = None
        self._block_batchnorm = None
        # public feature extractor
        self._block_publicFeature = None
        # create stage block
        #   (1) stage1 only have a belief map block
        if stage == 0:
            self._block_beliefMap = self.create_blockBeliefMap1()
        #   (2) stage2 ~ T
        else:
            # used for average pooling
            self._block_averagePool = nn.AvgPool2D(pool_size=8, strides=8)
            # used for belief map
            self._block_beliefMap = self.create_blockBeliefMap2_()
            # used for dependent feature extracting
            self._block_feature = nn.Conv2D(channels=32, kernel_size=5, strides=1, padding=2, activation='relu')
            self._block_batchnorm = nn.BatchNorm()
            # public feature extractor
            self._block_publicFeature = block_public_feature

    def forward(self, input_image, center_map, p_beliefMap):
        """
        Calculate forward.
        All params are in context gpu.
        :param input_image:
        :param center_map:
        :param p_beliefMap:
        :return:
        """
        # stage1
        if self._stage == 0:
            return self._block_beliefMap(input_image)
        # stage2 ~ T
        else:
            pool_center_map = self._block_averagePool(center_map)
            feature_map = self._block_batchnorm(self._block_feature(self._block_publicFeature(input_image)))
            concat_stage = nd.concat(pool_center_map, feature_map, p_beliefMap)
            return self._block_beliefMap(concat_stage)

    def create_blockBeliefMap1(self):
        """
        Belief map block for stage1
        :return: A sequential block which taken as part of input for 2nd stage
        """
        blockBeliefMap1 = nn.Sequential()
        with blockBeliefMap1.name_scope():
            # 1st conv layer
            blockBeliefMap1.add(nn.Conv2D(channels=128, kernel_size=9, strides=1, padding=4, activation='relu'))
            blockBeliefMap1.add(nn.BatchNorm())
            blockBeliefMap1.add(nn.MaxPool2D(pool_size=2, strides=2))
            # 2nd conv layer
            blockBeliefMap1.add(nn.Conv2D(channels=128, kernel_size=9, strides=1, padding=4, activation='relu'))
            blockBeliefMap1.add(nn.BatchNorm())
            blockBeliefMap1.add(nn.MaxPool2D(pool_size=2, strides=2))
            # 3rd conv layer
            blockBeliefMap1.add(nn.Conv2D(channels=128, kernel_size=9, strides=1, padding=4, activation='relu'))
            blockBeliefMap1.add(nn.BatchNorm())
            blockBeliefMap1.add(nn.MaxPool2D(pool_size=2, strides=2))
            # 4th conv layer
            blockBeliefMap1.add(nn.Conv2D(channels=32, kernel_size=5, strides=1, padding=2, activation='relu'))
            blockBeliefMap1.add(nn.BatchNorm())
            # 5th conv layer
            blockBeliefMap1.add(nn.Conv2D(channels=512, kernel_size=9, strides=1, padding=4, activation='relu'))
            blockBeliefMap1.add(nn.BatchNorm())
            # fc layer
            blockBeliefMap1.add(nn.Conv2D(channels=512, kernel_size=1, strides=1, padding=0, activation='relu'))
            blockBeliefMap1.add(nn.BatchNorm())
            blockBeliefMap1.add(nn.Conv2D(channels=self._keypoints_count, kernel_size=1, strides=1, padding=0, activation='relu'))
            blockBeliefMap1.add(nn.BatchNorm())
        return blockBeliefMap1

    def create_blockBeliefMap2_(self):
        """
        Belief map block for stage2 to stage T
        :return: A sequential block which taken as part of input for next stage
        """
        blockBeliefMap2_ = nn.Sequential()
        with blockBeliefMap2_.name_scope():
            # 1st conv layer
            blockBeliefMap2_.add(nn.Conv2D(channels=128, kernel_size=11, strides=1, padding=5, activation='relu'))
            blockBeliefMap2_.add(nn.BatchNorm())
            # 2nd conv laye
            blockBeliefMap2_.add(nn.Conv2D(channels=128, kernel_size=11, strides=1, padding=5, activation='relu'))
            blockBeliefMap2_.add(nn.BatchNorm())
            # 3rd conv layer
            blockBeliefMap2_.add(nn.Conv2D(channels=128, kernel_size=11, strides=1, padding=5, activation='relu'))
            blockBeliefMap2_.add(nn.BatchNorm())
            # fc layer
            blockBeliefMap2_.add(nn.Conv2D(channels=128, kernel_size=1, strides=1, padding=0, activation='relu'))
            blockBeliefMap2_.add(nn.BatchNorm())
            blockBeliefMap2_.add(nn.Conv2D(channels=self._keypoints_count, kernel_size=1, strides=1, padding=0, activation='relu'))
            blockBeliefMap2_.add(nn.BatchNorm())
        return blockBeliefMap2_


class CPM(nn.Block):
    """
    Convolutional Pose Machine
    """
    def __init__(self, network_name='CPM', stage_count=4, norm_image_size=(368, 368), belief_map_size=(46, 46), keypoints_count=24, **kwargs):
        super(CPM, self).__init__(**kwargs)
        # member variables
        self._keypoints_count = keypoints_count
        self._name = network_name
        self._norm_image_size = norm_image_size
        self._belief_map_size = belief_map_size
        #   used for public feature extractor
        self._block_public_feature = self.create_blockFeature()
        #   each stage of CPM
        self._block_stage = []
        # create model
        for stage in range(stage_count):
            self._block_stage.append(CPM_stage(network_name, keypoints_count, stage, self._block_public_feature))
            self.register_child(self._block_stage[stage])

    def forward(self, input_images, center_maps):
        """
        Calculate forward.
        All params are in context gpu.
        :param input_images:
        :param center_maps:
        :return:
        """
        # result
        result = []
        # calculate
        p_beliefMap = None
        for stage in range(len(self._block_stage)):
            p_beliefMap = self._block_stage[stage].forward(
                input_image=input_images,
                center_map=center_maps,
                p_beliefMap=p_beliefMap)
            result.append(p_beliefMap)
        return result

    def train(self, train_data, log_folder, params_folder, epochs, batch_size, ctx, init_lr, lr_step=5, lr_factor=0.1):
        """
        Train network.
        :param train_mode:
        :param train_data: Data and Label for training. Instance of tuple(dict).
        - valid_keys: valid keys of current category of clothes.
        - images: Instance of tuple. All images info of current category:
            - orig_images_id: Instance of list. (image_count)
            - orig_images_shape: Instance of np.array. (image_count, orig_h, orig_w)
            - orig_keypoints: Instance of np.array. (image_count, keypoints_count, 3)
            - norm_images: Instance of np.array. (image_count, 3, h, w)
            - belief_maps: Instance of np.array. (image_count, keypoints_count, h, w)
        - norm_centermap: Instance of np.array. (h, w)
        :param params_folder: Folder holds saved params.
        :param epochs:
        :param batch_size:
        :param ctx: Instance of list.
        :return:
        """
        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.StreamHandler(),
                                      logging.FileHandler(log_folder + 'train_' + self._name + '_batch_' + str(epochs) + '_' + str(batch_size))])

        # 1. check params files and get last epoch and batch
        epoch_index, batch_index, file = self.utils_params_file(batch_size, 'check', params_folder)
        # (1) begin a new training
        if epoch_index == -1 and batch_index == -1:
            logging.info("No params files detected. Begin a new training.")
            self.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)
            epoch_index = 0
            batch_index = 0
        # (2) resume training from params file
        else:
            logging.info("Params file '%s' detected. Last (epoch, batch): (%d, %d). Resuming training." % (file, epoch_index, batch_index))
            self.collect_params().load(params_folder + file, ctx=ctx)
            batch_index += 1
        # 2. train
        # (1) trainer and loss function for total training mode
        model_trainer = trainer.Trainer(self.collect_params(), 'sgd',
                                        {'learning_rate': init_lr, 'momentum': 0.9, 'wd': 5e-4})
        #loss_function = loss.SoftmaxCrossEntropyLoss(sparse_label=False)
        loss_function = loss.SigmoidBinaryCrossEntropyLoss()
        # (2) train each epoch and batch
        for e in range(epoch_index, epochs):
            if e != epoch_index: batch_index = 0
            # 1> set learning rate
            model_trainer.set_learning_rate(init_lr * pow(lr_factor, int(e / lr_step)))
            if e % lr_step == 0:
                logging.info('Learning rate now is set to be %.6f' % model_trainer.learning_rate)
            # 2> train batch
            while True:
                # (1) get data
                _, _, orig_images_shape_batch, orig_keypoints_batch, norm_images_batch, norm_center_maps_batch, belief_maps_batch, _ = \
                    train_data.get_batch_data(if_data_aug=True, loss_mode='softmax', batch_index=batch_index, batch_size=batch_size)
                if norm_images_batch is None and norm_center_maps_batch is None and belief_maps_batch is None: break
                # (2) split data into multiple GPU
                norm_images_batch_LIST = split_and_load(norm_images_batch, ctx_list=ctx)
                norm_center_maps_batch_LIST = split_and_load(norm_center_maps_batch, ctx_list=ctx)
                belief_maps_batch_LIST = split_and_load(belief_maps_batch, ctx_list=ctx)
                #-------------------------------------------------------------------------------------------------------
                # (3) train total
                pred_beliefMaps_batch = []
                # 1> record auto grad
                with autograd.record():
                    # 1.initiate gpu losses
                    gpu_losses = []
                    # 2.calculate losses on each gpu of each stage
                    for norm_images_batch, norm_center_maps_batch, belief_maps_batch in zip(norm_images_batch_LIST,
                                                                                            norm_center_maps_batch_LIST,
                                                                                            belief_maps_batch_LIST):
                        # (1) initiate current gpu loss
                        current_gpu_loss = None
                        # (2) network forward
                        pred_beliefMaps = self.forward(input_images=norm_images_batch, center_maps=norm_center_maps_batch)
                        for p_b in pred_beliefMaps[-1].asnumpy(): pred_beliefMaps_batch.append(p_b)
                        # (3) shape groud-truth belief maps to use softmax loss
                        shaped_gt_beliefMaps = nd.reshape(belief_maps_batch,
                                                          shape=(belief_maps_batch.shape[0], belief_maps_batch.shape[1],
                                                                 belief_maps_batch.shape[2] * belief_maps_batch.shape[3]))
                        # (4) calculate each and every stage loss on current gpu
                        for stage in range(len(self._block_stage)):
                            # 1> shape predicted belief map of current stage
                            shaped_pred_beliefMap = nd.reshape(pred_beliefMaps[stage],
                                                               shape=(pred_beliefMaps[stage].shape[0],
                                                                      pred_beliefMaps[stage].shape[1],
                                                                      pred_beliefMaps[stage].shape[2] *
                                                                      pred_beliefMaps[stage].shape[3]))
                            # 2> calculate current stage loss on current gpu
                            current_loss = loss_function(shaped_pred_beliefMap, shaped_gt_beliefMaps)
                            # 3> summary
                            current_gpu_loss = current_loss if current_gpu_loss is None else (current_gpu_loss + current_loss)
                        # (5) append & save
                        gpu_losses.append(current_gpu_loss)
                # 3> backward and update
                for gpu_loss in gpu_losses:
                    gpu_loss.backward()
                model_trainer.step(batch_size)
                nd.waitall()
                # 4> calculate batch average loss
                batch_loss = sum([nd.sum(gpu_loss).asscalar() for gpu_loss in gpu_losses]) / (batch_size * len(self._block_stage))
                NE = self.calculate_error(valid_keys=utils.keypoints_order[train_data.category],
                                          category=train_data.category,
                                          predicted_keypoints=self.transform_beliefMaps_into_origKeypoints(
                                              predicted_beliefMaps=np.array(pred_beliefMaps_batch),
                                              orig_images_shape=orig_images_shape_batch),
                                          orig_keypoints=np.array(orig_keypoints_batch))
                # 5> print
                logging.info("Epoch[%d]-Batch[%d] lr: %f. Average loss: %f. NE:%.2f%%" % (e, batch_index, model_trainer.learning_rate, batch_loss, NE*100))
                #-------------------------------------------------------------------------------------------------------
                # (4) save params with batch info (batch_size, batch_index)
                params_file = self.utils_params_file(operation='generate', batch_size=batch_size, epoch_index=e, batch_index=batch_index)
                params_old_file = self.utils_params_file(operation='generate', batch_size=batch_size,
                                                         epoch_index=e, batch_index=batch_index - 1, batches=train_data.calc_batches_count(batch_size))
                self.collect_params().save(params_folder + params_file)
                if os.path.exists(params_folder + params_old_file): os.remove(params_folder + params_old_file)
                batch_index += 1
        # 3.finish
        logging.info("Training completed.")

    def predict(self, test_data, log_folder, params_folder, folder_holds_rst_pics, path_to_save_file, batch_size, gpu_id):
        """
        Predict result with single GPU.
        :param test_data: Data for predicting. Instance of tuple.
            - orig_images_id:
            - orig_images_shape:
            - norm_images:
            - norm_centermap:
        :param params_folder:
        :param batch_size:
        :param gpu_id:
        :return:
        """
        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.StreamHandler(),
                                      logging.FileHandler(log_folder + 'test_' + self._name + '_batch_' + str(batch_size))])

        # 0.result
        predicted_result = {'images_id': [], 'keypoints': []}
        orig_keypoints = []
        # 1.check params file
        params_file = self.utils_params_file(batch_size, 'check', params_folder)[2]
        if params_file is None:
            logging.info("No params files detected. Please train first.")
            return
        # 2.load params
        logging.info("Loading params file from detected file '%s'..." % params_file)
        self.collect_params().load(params_folder + params_file, ctx=mx.gpu(gpu_id))
        # 3.predict
        batch_index = 0
        while True:
            # (1) get data
            orig_images_id_batch, orig_images_batch, orig_images_shape_batch, orig_keypoints_batch, norm_images_batch, norm_centermap_batch, _, _ = \
                test_data.get_batch_data(if_data_aug=False, loss_mode='softmax', batch_size=batch_size, batch_index=batch_index)
            if orig_images_id_batch is None and orig_images_batch is None and orig_images_shape_batch is None and \
                    norm_images_batch is None and norm_centermap_batch is None: break
            # (2) predict
            predicted_beliefMaps = self.forward(input_images=nd.array(norm_images_batch, ctx=mx.gpu(gpu_id)),
                                                center_maps=nd.array(norm_centermap_batch, ctx=mx.gpu(gpu_id)))[-1]
            #predicted_beliefMaps = None
            #for pred_BMs in _predicted_beliefMaps:
            #    predicted_beliefMaps = pred_BMs if predicted_beliefMaps is None else (predicted_beliefMaps + pred_BMs)
            # (3) transfer predicted belief maps into original key points
            predicted_orig_keypoints_batch = self.transform_beliefMaps_into_origKeypoints(
                predicted_beliefMaps=predicted_beliefMaps.asnumpy(),
                orig_images_shape=orig_images_shape_batch)
            # (4) save
            for orig_image, pred_orig_keypoint, orig_images_id in zip(orig_images_batch, predicted_orig_keypoints_batch, orig_images_id_batch):
                # 1> predicted result
                predicted_result['images_id'].append(orig_images_id)
                predicted_result['keypoints'].append(pred_orig_keypoint)
                # 2> images with predicted result
                utils.show_image_and_keypoints(image=orig_image,
                                               keypoints=pred_orig_keypoint,
                                               image_name=os.path.splitext(orig_images_id.split('/')[-1])[0],
                                               save_folder=folder_holds_rst_pics)
            # (5) if original key points is not None, calculate NE
            if orig_keypoints_batch is not None:
                # 1> save original key points
                for orig_kp_batch in orig_keypoints_batch: orig_keypoints.append(orig_kp_batch)
                # 2> calculate NE
                NE = self.calculate_error(valid_keys=utils.keypoints_order[test_data.category],
                                          category=test_data.category,
                                          predicted_keypoints=predicted_orig_keypoints_batch,
                                          orig_keypoints=orig_keypoints_batch)
                logging.info("Batch[%d] predicting completed. NE: %f%%." % (batch_index, NE*100))
            else:
                logging.info("Batch[%d] predicting completed." % batch_index)
            batch_index += 1
        # 4.finish
        predicted_file = open(path_to_save_file, 'wb')
        pickle.dump(predicted_result, predicted_file)
        predicted_file.close()
        # 5.calculate total NE
        if len(orig_keypoints) != 0:
            NE = self.calculate_error(valid_keys=utils.keypoints_order[test_data.category],
                                      category=test_data.category,
                                      predicted_keypoints=predicted_result['keypoints'],
                                      orig_keypoints=orig_keypoints)
            logging.info("Predicting completed. NE: %.2f%%." % (NE*100))
        else:
            logging.info("Predicting completed.")

    def create_blockFeature(self):
        """
        Feature extractor block used in CPM
        :return: A sequential block which taken as part of the input for each except 1st stage
        """
        blockFeature = nn.Sequential()
        with blockFeature.name_scope():
            # 1st conv layer
            blockFeature.add(nn.Conv2D(channels=128, kernel_size=9, strides=1, padding=4, activation='relu'))
            blockFeature.add(nn.BatchNorm())
            blockFeature.add(nn.MaxPool2D(pool_size=2, strides=2))
            # 2nd conv layer
            blockFeature.add(nn.Conv2D(channels=128, kernel_size=9, strides=1, padding=4, activation='relu'))
            blockFeature.add(nn.BatchNorm())
            blockFeature.add(nn.MaxPool2D(pool_size=2, strides=2))
            # 3rd conv layer
            blockFeature.add(nn.Conv2D(channels=128, kernel_size=9, strides=1, padding=4, activation='relu'))
            blockFeature.add(nn.BatchNorm())
            blockFeature.add(nn.MaxPool2D(pool_size=2, strides=2))
        return blockFeature

    def transform_beliefMaps_into_origKeypoints(self, predicted_beliefMaps, orig_images_shape):
        """
        Transform belief maps into original key points.
        All variables are in context cpu.
            - if the maximum of belief map is lower than the given threshold, the key point is predicted to not exist.
            - find the location of the maximum of belief map and project it to the original image.
        :param predicted_beliefMaps: Instance of numpy.array. (batch_size, keypoints_count, size1, size2)
        :param orig_images_shape: Instance of numpy.array. (batch_size, 2)
        :return: original key points in original images. Instance of numpy.array. (batch_size, keypoints_count, 2)
        """
        def find_2D_argmax(array):
            """
            Find location of the maximum of a 2-D array.
            :param array: Instance of numpy.array.
            :return: (location1, location2, maximum)
            """
            max_cols = np.argmax(array, axis=1)
            max_row = np.argmax(np.array([array[row][max_cols[row]] for row in range(array.shape[0])]))
            return max_row, max_cols[max_row], array[max_row][max_cols[max_row]]
        # 1.initialize result
        orig_keypoints = []
        # 2.calculate
        for image_index in range(predicted_beliefMaps.shape[0]):
            orig_image_keypoints = []
            # (1) ratio
            ratio_h = orig_images_shape[image_index][0]/self._belief_map_size[0]
            ratio_w = orig_images_shape[image_index][1]/self._belief_map_size[1]
            # (2) result
            for key_index in range(predicted_beliefMaps.shape[1]):
                h_beliefMap, w_beliefMap, maximum = find_2D_argmax(predicted_beliefMaps[image_index][key_index])
                orig_image_keypoints.append([w_beliefMap*ratio_w, h_beliefMap*ratio_h])
            orig_keypoints.append(orig_image_keypoints)
        # 3.return
        return np.array(orig_keypoints)

    def calculate_error(self, valid_keys, category, predicted_keypoints, orig_keypoints):
        """
        Calculate error by given key points.
        :param predicted_keypoints: (batch_size, keypoints_count, 2)
        :param orig_keypoints: (batch_size, keypoints_count, 3)
        :return:
        """
        # normalization key
        norm_keys = {'blouse':  ['armpit_right', 'armpit_left'],
                     'outwear': ['armpit_right', 'armpit_left'],
                     'dress':   ['armpit_right', 'armpit_left'],
                     'trousers':['waistband_left', 'waistband_right'],
                     'skirt':   ['waistband_left', 'waistband_right']}[category]

        # 1. initiate error
        numerator = 0
        denominator = 0
        # 2. calculate each image
        key_index0 = valid_keys.index(norm_keys[0])
        key_index1 = valid_keys.index(norm_keys[1])
        for pred_kp, orig_kp in zip(predicted_keypoints, orig_keypoints):
            # (1) calculate normalization distance
            norm_distance = np.sqrt(np.sum((orig_kp[key_index0][:2] - orig_kp[key_index1][:2])**2))
            if norm_distance < 1: continue
            # (2) calculate numerator and denominator of current image
            numerator += np.sum([(np.sqrt(np.sum((pred_kp[key_index][:2] - orig_kp[key_index][:2])**2)) if orig_kp[key_index][2] == 1 else 0)
                         for key_index in range(len(valid_keys))]) / norm_distance
            denominator += np.sum([(1 if orig_kp[key_index][2] == 1 else 0) for key_index in range(len(valid_keys))])
        # 3. calculate NE
        if denominator == 0:
            return -1
        else:
            return numerator / denominator

    def utils_params_file(self, batch_size, operation, params_folder=None, epoch_index=None, batch_index=None, batches=None):
        """
        Utility function for param files.
        :param batch_index:
        :param batch_size:
        :param params_folder:
        :return:
        """
        # 1.check file existence
        if operation == 'check':
            # file exists
            for root, _, files in os.walk(params_folder):
                for file in files:
                    # 1. check extension
                    if os.path.splitext(file)[1] != '.params': continue
                    # 2. check file name, file_split should be (name, 'batch', epoch_index, batch_size, batch_index)
                    file_split = str(os.path.splitext(file)[0]).split('_')
                    #   (1) check splitting length
                    if len(file_split) != len(self._name.split('_')) + 4: continue
                    #   (2) check self_name
                    name_ok = True
                    self_name_split = self._name.split('_')
                    for index in range(len(self_name_split)):
                        if self_name_split[index] != file_split[index]:
                            name_ok = False
                            break
                    if not name_ok: continue
                    #   (3) check 'batch'
                    if file_split[len(self_name_split)] != 'batch': continue
                    #   (4) check batch_size
                    try:
                        int(file_split[len(self_name_split) + 1])
                        int(file_split[len(self_name_split) + 2])
                        int(file_split[len(self_name_split) + 3])
                    except:
                        continue
                    if int(file_split[len(self_name_split) + 2]) != batch_size: continue
                    # check pass, get batch_index and epoch_index
                    epoch_index = int(file_split[len(self_name_split) + 1])
                    batch_index = int(file_split[len(self_name_split) + 3])
                    return epoch_index, batch_index, file
            return -1, -1, None
        # 2.generate file name
        elif operation == 'generate':
            # if batch_index < 0, then go to last epoch
            if batch_index < 0:
                return self._name + '_' + 'batch' + '_' + str(epoch_index-1) + '_' + str(batch_size) + '_' + str(batches-1) + '.params'
            # else
            else:
                return self._name + '_' + 'batch' + '_' + str(epoch_index) + '_' + str(batch_size) + '_' + str(batch_index) + '.params'