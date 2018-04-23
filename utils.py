"""
Utility functions for project.
"""

import os
import xlrd
import xlwt
import math
import pickle
import random
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.transform import resize
from skimage.transform import rotate


########################################################################################################################
# DO NOT MODIFY !!!
keypoints_order = {
    'blouse':   ['neckline_right', 'armpit_right', 'cuff_left_out', 'neckline_left', 'cuff_left_in', 'top_hem_right', 'cuff_right_out',
                 'center_front', 'shoulder_right', 'shoulder_left', 'cuff_right_in', 'top_hem_left', 'armpit_left'],
    'outwear':  ['shoulder_left', 'cuff_left_in', 'top_hem_left', 'shoulder_right', 'neckline_left', 'cuff_left_out',
                 'cuff_right_in', 'armpit_right', 'top_hem_right', 'neckline_right', 'armpit_left', 'cuff_right_out', 'waistline_left', 'waistline_right'],
    'dress':    ['center_front', 'armpit_right', 'waistline_right', 'neckline_left', 'cuff_right_in', 'shoulder_right', 'neckline_right',
                 'hemline_left', 'shoulder_left', 'cuff_right_out', 'hemline_right', 'cuff_left_in', 'cuff_left_out', 'armpit_left', 'waistline_left'],
    'trousers': ['bottom_left_in', 'bottom_left_out', 'crotch', 'waistband_left', 'bottom_right_in', 'waistband_right', 'bottom_right_out'],
    'skirt':    ['hemline_left', 'waistband_right', 'hemline_right', 'waistband_left']
}
########################################################################################################################



class dataLoader:
    """
    Load all the data into memory. Instance of tuple. Each key of the tuple is the corresponding category name. Each item
    in the tuple is instance of tuple:
    --------------------------------------------------------------------------------------------------------------------
    For training data:
    - images: Instance of tuple. All images info of current category:
        - orig_images_id: Instance of list. (image_count)
        - orig_images_shape: Instance of np.array. (image_count, orig_h, orig_w)
        - orig_keypoints: Instance of np.array. (image_count, keypoints_count, 3)
        - norm_images: Instance of np.array. (image_count, 3, h, w)
        - belief_maps: Instance of np.array. (image_count, keypoints_count, h, w)
    - norm_centermap: Instance of np.array. (h, w)
    --------------------------------------------------------------------------------------------------------------------
    For testing data:
    - images: Instance of tuple. All images info of current category.
        - orig_images_id: Instance of list. (image_count)
        - orig_images_shape: Instance of np.array. (image_count)
        - norm_images: Instance of np.array. (image_count, 3, h, w)
    - norm_centermap: Instance of np.array. (h, w)
    --------------------------------------------------------------------------------------------------------------------
    """

    def __init__(self, category, path_to_excel_file, images_prefix, norm_image_size, belief_map_size):
        self._category = category
        self._images_prefix = images_prefix
        self._norm_image_size = norm_image_size
        self._belief_map_size = belief_map_size
        # 1. load excel file    # orig_images_id (orig_keypoints)
        self._data = self.read_excel(path_to_excel_file, category)
        # 2. generate center map
        self._data['norm_centermap'] = self.generate_gaussian_map(image_size=norm_image_size,
                                                                  center_location=(norm_image_size[0]/2, norm_image_size[1]/2))

    @property
    def category(self):
        return self._category

    def get_batch_data(self, if_data_aug, loss_mode, batch_size, batch_index):
        """
        :param batch_index:
        :param batch_size:
        :return:
            - orig_images_id                    (predict)
            - orig_images                       (predict)
            - orig_images_shape                 (predict)
            - orig_keypoints                    (predict)
            - norm_images               (train) (predict)
            - norm_centermap            (train) (predict)
            - belief_maps               (train)
            - loss_mask (for l2loss)    (train)
        """
        images_count = len(self._data['images']['orig_images_id'])
        # 1.calculate start and finish indexes
        s_index = batch_size * batch_index
        e_index = batch_size * (batch_index + 1)
        if e_index > images_count: e_index = images_count
        # 2.indexes exceed
        if s_index >= e_index:
            print("Indexes exceed. Abort.")
            return None, None, None, None, None, None, None, None
        # 3.return batch data
        else:
            if e_index - s_index < batch_size: s_index = e_index - batch_size
            # (0) does have ground-truth ?
            mode = 'train' if 'orig_keypoints' in self._data['images'].keys() else 'refer'
            # (1)if load batch data from file and save to self.
            for index in range(s_index, e_index):
                if str(index) not in self._data['images']['norm_images'].keys():
                    # 1> load original images
                    orig_images = self.get_orig_images(images_prefix=self._images_prefix, _range=(index, index + 1))
                    # 2> normalize images and get original images shape
                    norm_images, orig_images_shape = \
                        self.normalize_data(orig_images=orig_images, norm_image_size=self._norm_image_size)
                    self._data['images']['norm_images'].update({str(index): norm_images[0]})
                    self._data['images']['orig_images_shape'].update({str(index): orig_images_shape[0]})
                    # 3> generate belief map for training
                    if mode == 'train':
                        belief_maps = self.generate_belief_maps(loss_mode=loss_mode,
                                                                orig_images_shape=orig_images_shape,
                                                                orig_keypoints=np.array([self._data['images']['orig_keypoints'][index]]),
                                                                beliefmap_size=self._belief_map_size)
                        self._data['images']['belief_maps'].update({str(index): belief_maps[0]})
            # (2)else self contains data of current batch, return directly
            r_orig_images_id = self._data['images']['orig_images_id'][s_index:e_index]
            r_orig_images = [imread(self._images_prefix + self._data['images']['orig_images_id'][index]) for index in range(s_index, e_index)]
            r_orig_images_shape = [self._data['images']['orig_images_shape'][str(index)] for index in range(s_index, e_index)]
            r_orig_keypoints = self._data['images']['orig_keypoints'][s_index:e_index] if mode == 'train' else None
            r_norm_images = [self._data['images']['norm_images'][str(index)] for index in range(s_index, e_index)]
            r_norm_centermap = np.array([[self._data['norm_centermap']] for _ in range(s_index, e_index)])
            r_belief_maps = [self._data['images']['belief_maps'][str(index)] for index in range(s_index, e_index)] if mode == 'train' else None
            r_loss_mask = [[0 if self._data['images']['orig_keypoints'][image_index][key_index][0] == -1 else 1
                     for key_index in range(len(keypoints_order[self._category]))]
                    for image_index in range(s_index, e_index)] if mode == 'train' else None
            # ----------------------------------------------------------------------------------------------------------
            # DATA AUGMENTATION
            if if_data_aug == True:
                r_norm_images = []
                r_belief_maps = []
                r_orig_keypoints = []
                for index in range(s_index, e_index):
                    # 1. original image and key points
                    orig_image = imread(self._images_prefix + self._data['images']['orig_images_id'][index])
                    orig_key_points = self._data['images']['orig_keypoints'][index]
                    # 2. augmentation
                    t_image, t_keypoints, flag = augmentation(orig_image, orig_key_points)
                    # 3. append & save
                    # (1) original image, fetch data from self data
                    if flag == 0:
                        r_norm_images.append(self._data['images']['norm_images'][str(index)])
                        r_belief_maps.append(self._data['images']['belief_maps'][str(index)])
                    # (2) rotated and scaled image
                    else:
                        # 1> generate normalized image for transferred image
                        t_norm_image, _ = \
                            self.normalize_data(orig_images=[t_image], norm_image_size=self._norm_image_size)
                        r_norm_images.append(t_norm_image[0])
                        # 2> generate belief maps for transferred image
                        r_belief_maps.append(self.generate_belief_maps(loss_mode=loss_mode,
                                                                       orig_images_shape=np.array([t_image.shape[:2]]),
                                                                       orig_keypoints=np.array([t_keypoints]),
                                                                       beliefmap_size=self._belief_map_size)[0])
                    r_orig_keypoints.append(t_keypoints)
            #-----------------------------------------------------------------------------------------------------------
            return r_orig_images_id, \
                   r_orig_images, \
                   r_orig_images_shape, \
                   r_orig_keypoints, \
                   r_norm_images, \
                   r_norm_centermap, \
                   r_belief_maps, \
                   r_loss_mask

    def calc_batches_count(self, batch_size):
        """
        :return:
        """
        return int(math.ceil(len(self._data['images']['orig_images_id'])/batch_size))

    @staticmethod
    def read_excel(path_to_excel_file, category):
        """
        Return format.
        :param path_to_excel_file:
        :param category:
        :return:
        - for training:
            - images:
                - orig_images_id        xxx
                - orig_images_shape
                - orig_keypoints        xxx
                - norm_images
                - belief_maps
            - norm_centermap
        - for refering:
            - images:
                - orig_images_id        xxx
                - orig_images_shape
                - norm_images
            - norm_centermap
        """
        # 1.Open file and getthe first sheet
        workbook = xlrd.open_workbook(path_to_excel_file)
        sheet = workbook.sheet_by_index(0)
        # which mode? if excel has just 2 columns -> 'refer', else -> 'train'.
        mode = 'refer' if sheet.ncols == 2 else 'train'
        # 2.for training
        if mode == 'train':
            # (1).load original data
            orig_data = []
            for row in range(1, sheet.nrows):
                if sheet.cell_value(row, 1) != category: continue
                data_row = {
                    'image_id': sheet.cell_value(row, 0),
                    'image_category': sheet.cell_value(row, 1),
                    'neckline_left': sheet.cell_value(row, 2),
                    'neckline_right': sheet.cell_value(row, 3),
                    'center_front': sheet.cell_value(row, 4),
                    'shoulder_left': sheet.cell_value(row, 5),
                    'shoulder_right': sheet.cell_value(row, 6),
                    'armpit_left': sheet.cell_value(row, 7),
                    'armpit_right': sheet.cell_value(row, 8),
                    'waistline_left': sheet.cell_value(row, 9),
                    'waistline_right': sheet.cell_value(row, 10),
                    'cuff_left_in': sheet.cell_value(row, 11),
                    'cuff_left_out': sheet.cell_value(row, 12),
                    'cuff_right_in': sheet.cell_value(row, 13),
                    'cuff_right_out': sheet.cell_value(row, 14),
                    'top_hem_left': sheet.cell_value(row, 15),
                    'top_hem_right': sheet.cell_value(row, 16),
                    'waistband_left': sheet.cell_value(row, 17),
                    'waistband_right': sheet.cell_value(row, 18),
                    'hemline_left': sheet.cell_value(row, 19),
                    'hemline_right': sheet.cell_value(row, 20),
                    'crotch': sheet.cell_value(row, 21),
                    'bottom_left_in': sheet.cell_value(row, 22),
                    'bottom_left_out': sheet.cell_value(row, 23),
                    'bottom_right_in': sheet.cell_value(row, 24),
                    'bottom_right_out': sheet.cell_value(row, 25)
                }
                orig_data.append(data_row)
            # (2).initiate returning data
            data = {
                'images': {'orig_images_id': [], 'orig_images_shape': {}, 'orig_keypoints': [],
                           'norm_images': {}, 'belief_maps': {}},
                'norm_centermap': []
            }
            # (3).load original images id and images key points
            for record in orig_data:
                # 1> save image id
                data['images']['orig_images_id'].append(record['image_id'])
                # 2> save original images key points
                current_image_keypoints = []
                for key_name in keypoints_order[category]:
                    loc1, loc2, visible = record[key_name].split('_')
                    current_image_keypoints.append([int(loc1), int(loc2), int(visible)])
                data['images']['orig_keypoints'].append(np.array(current_image_keypoints))
            data['images']['orig_keypoints'] = np.array(data['images']['orig_keypoints'])
            return data
        elif mode == 'refer':
            # (1).load original data
            orig_data = []
            for row in range(1, sheet.nrows):
                if sheet.cell_value(row, 1) != category: continue
                data_row = {
                    'image_id': sheet.cell_value(row, 0),
                    'image_category': sheet.cell_value(row, 1),
                }
                orig_data.append(data_row)
            # (2).initiate returning data
            data = {
                'images': {'orig_images_id': [], 'orig_images_shape': {}, 'norm_images': {}},
                'norm_centermap': []
            }
            # (3).load original images id
            for record in orig_data:
                # save image id
                data['images']['orig_images_id'].append(record['image_id'])
            return data

    def get_orig_images(self, images_prefix, _range):
        """
        Get original images.
        :return: original images. Instance of list. (images_count, h, w, 3)
        """
        return [imread(images_prefix + self._data['images']['orig_images_id'][image_index])
                for image_index in range(_range[0], _range[1])]

    def normalize_data(self, orig_images, norm_image_size):
        """
        Normalize orignal images and key points. Channel should be changed into first.
        :param orig_images: Original images. Instance of list. (image_count, h, w, 3)
        :param norm_image_size: Normalized image size. Instance of tuple. (h, w)
        :param verbose: if show information
        :return:
        - Normalized images. Instance of numpy.array.
        - Original images shape. Instance of numpy.array.
        """
        # 1. original images shape
        orig_images_shape = np.array([
            [orig_images[image_index].shape[0],
             orig_images[image_index].shape[1]] for image_index in range(len(orig_images))])
        # 2. normalize original images
        #   (1) resize images
        for image_index in range(len(orig_images)):
            orig_images[image_index] = resize(orig_images[image_index], norm_image_size, mode='constant')
        #   (2) exchange channel
        for image_index in range(len(orig_images)):
            orig_images[image_index] = (np.array(
                [orig_images[image_index][:, :, 0] - np.average(orig_images[image_index][:, :, 0]),
                 orig_images[image_index][:, :, 1] - np.average(orig_images[image_index][:, :, 1]),
                 orig_images[image_index][:, :, 2] - np.average(orig_images[image_index][:, :, 2])])/1.00).astype('float32')
        return np.array(orig_images), orig_images_shape

    def generate_belief_maps(self, loss_mode, orig_images_shape, orig_keypoints, beliefmap_size):
        """
        Generate belief maps of beliefmap_size.
        :param orig_images_shape: Original images. Instance of numpy.array. (image_count, orig_h, orig_w)
        :param orig_keypoints: Orignal key points. Instance of numpy.array. (image_count, keypoints_count, 3)
        :param beliefmap_size: Instance of tuple. (h, w)
        :param verbose
        :return: belief maps. Instance of numpy.array. (image_count, keypoints_count, size1, size2)
        """
        assert loss_mode in ['softmax', 'l2'], 'Wrong loss mode.'
        belief_maps = []
        for image_index in range(orig_keypoints.shape[0]):
            belief_map = []
            for key_index in range(orig_keypoints.shape[1]):
                # 1.if has the key points, generate gaussian_map
                if (orig_keypoints[image_index][key_index][0] != -1) and (orig_keypoints[image_index][key_index][1] != -1):
                    """
                    if loss_mode == 'softmax':
                        belief_map.append(self.generate_prob_map(
                            beliefmap_size,
                            (orig_keypoints[image_index][key_index][1] / orig_images_shape[image_index][0] *
                             beliefmap_size[0],
                             orig_keypoints[image_index][key_index][0] / orig_images_shape[image_index][1] *
                             beliefmap_size[1])))
                    else:
                        belief_map.append(self.generate_gaussian_map(
                            beliefmap_size,
                            (orig_keypoints[image_index][key_index][1] / orig_images_shape[image_index][0] *
                             beliefmap_size[0],
                             orig_keypoints[image_index][key_index][0] / orig_images_shape[image_index][1] *
                             beliefmap_size[1])))
                    """
                    belief_map.append(self.generate_gaussian_map(
                        beliefmap_size,
                        (orig_keypoints[image_index][key_index][1] / orig_images_shape[image_index][0] *
                         beliefmap_size[0],
                         orig_keypoints[image_index][key_index][0] / orig_images_shape[image_index][1] *
                         beliefmap_size[1])))
                # 2.if doesn't have the key points, generate zero
                else:
                    belief_map.append(np.zeros(shape=beliefmap_size))
            belief_maps.append(belief_map)
        return np.array(belief_maps)

    def generate_gaussian_map(self, image_size, center_location):
        """
        Generate Gaussian map of image_size with a 2-D Gaussian function at center_location
        :param image_size: Instance of tuple. (h, w)
        :param center_location: Instance of tuple. (h_coor, w_coor)
        :return: Instance of numpy.array. (h, w)
        """
        center_x = center_location[0]
        center_y = center_location[1]
        #R = np.sqrt(center_x ** 2 + center_y ** 2)
        R = np.sqrt(image_size[0]**2 + image_size[1]**2)/70
        gaussian_map = np.array(
            [[np.exp(-0.5 * np.sqrt((j - center_x) ** 2 + (k - center_y) ** 2) / R)
              for k in range(image_size[1])]
             for j in range(image_size[0])])
        return gaussian_map

    def generate_prob_map(self, image_size, center_location):
        """
        Generate probability map of image_size with a unique 1 in the center of the map and others 0
        :param image_size: Instance of tuple. (h, w)
        :param center_location: Instance of tuple. (h_coor, w_coor)
        :return: Instance of numpy.array. (h, w)
        """
        center_x = int(round(center_location[0]))
        center_y = int(round(center_location[1]))
        if center_x >= image_size[0]: center_x = image_size[0] - 1
        if center_y >= image_size[1]: center_y = image_size[1] - 1
        prob_map = np.zeros(shape=image_size)
        prob_map[center_x][center_y] = 1
        return prob_map


def augmentation(image, orig_key_points):
    """
    Randomly rotate, scale or do nothing
    :param image:
    :param orig_key_points:
    :return:
    """
    which = random.randint(0, 10)
    # 0.rotate
    if which in [0, 1, 2]:
        t_image, t_keypoints = augmentation_rotate(image, orig_key_points)
        flag = 1
    # 1.scale
    elif which in [3, 4, 5, 6, 7, 8]:
        t_image, t_keypoints = augmentation_scale(image, orig_key_points)
        flag = 2
    # 2.keep
    else:
        t_image = image
        t_keypoints = orig_key_points
        flag = 0
    # return
    if t_image is None and t_keypoints is None:
        return image, orig_key_points, 0
    else:
        return t_image, t_keypoints, flag


def augmentation_rotate(image, orig_key_points):
    """
    Rotate image for data augmentation but make sure the transferred key point still in the transferred image
    :param image: Instance of Image. (h, w, 3)
    :param orig_key_points: Instance of np.array. (keypoints_count, 3)
    :return:
    """
    # if any of original key points is around image center, abort rotating
    image_center = np.array([image.shape[1]/2, image.shape[0]/2])
    for orig_key_point in orig_key_points:
        if np.sqrt(np.sum((orig_key_point[:2] - image_center)**2)) < 1:
            return image, orig_key_points
    # 1. rotating angle
    rnd = random.random()
    angle = rnd*60 - 30
    # 2. rotate images
    rotate_image = rotate(image=image, angle=angle, mode='constant')
    # 3. calculate rotate_key_point
    #-------------------------------------------------------------------------------------------------------------------
    # (1) transformation relationship between rotated point and original which takes image center as the coordinate origin
    # which positive direction of x axis and y axis is to the right and up, respectively.
    #     x_rotated = r * cos(angle)
    #     y_rotated = r * sin(angle)
    # which r = sqrt(x^2 + y^2) and angle in degrees in counter-clockwise direction
    def orig_to_rotated(point, angle):
        r = math.sqrt(point[0]**2 + point[1]**2)
        # 1. get original angle
        abs_angle = math.fabs(math.asin(point[1]/r))
        # (1) 1st quadrant
        if point[0] >= 0 and point[1] >= 0:
            orig_angle = abs_angle
        # (2) 2nd quadrant
        elif point[0] <= 0 and point[1] >= 0:
            orig_angle = math.pi - abs_angle
        # (3) 3rd quadrant
        elif point[0] <= 0 and point[1] <= 0:
            orig_angle = math.pi + abs_angle
        # (4) 4th quadrant
        else:
            orig_angle = -abs_angle
        return [r*math.cos(math.radians(angle)+orig_angle), r*math.sin(math.radians(angle)+orig_angle)]
    # (2) transformation relationship between point which takes image center as the corrdinate origin (which positive
    # direction of x axis and y axis is to the right and down, respectively) and point which takes top left corner as
    # the coordinate origin (which positive direction of x axis and y axis is to the right and up, respectively)
    def center_to_topleft(point, image_shape):
        height = image_shape[0]
        width = image_shape[1]
        return [point[0] + width/2, height/2 - point[1]]
    # (3) transformation relationship between point which takes top left corner as the coordinate origin (which positive
    # direction of x axis and y axis is to the right and down, respectively) and point which takes image center as the
    # coordinate origin (which positive direction of x axis and y axis is to the right and up, respectively)
    def topleft_to_center(point, image_shape):
        height = image_shape[0]
        width = image_shape[1]
        return [point[0] - width/2, height/2 - point[1]]
    #-------------------------------------------------------------------------------------------------------------------
    rotate_key_locations = [(center_to_topleft(
        point=orig_to_rotated(
            point=topleft_to_center(point=orig_key_point[:2],
                                    image_shape=image.shape[:2]),
            angle=angle),
        image_shape=image.shape[:2])
                             if orig_key_point[0] != -1 and orig_key_point[1] != -1 else [-1, -1])
                            for orig_key_point in orig_key_points]
    rotate_keypoints = [[rotate_key_location[0], rotate_key_location[1], orig_key_point[2]]
                        for rotate_key_location, orig_key_point in zip(rotate_key_locations, orig_key_points)]
    # 4. return
    # (1) if key point of transferred image out of boundary, abort rotation result and return None
    for rotate_key_location in rotate_keypoints:
        if rotate_key_location[0] >= rotate_image.shape[0] or rotate_key_location[1] >= rotate_image.shape[1] \
                or rotate_key_location[0] < 0 or rotate_key_location[1] < 0:
            return None, None
    # (2) return transferred result otherwise
    return rotate_image, rotate_keypoints


def augmentation_scale(image, orig_key_points):
    """
    Zoom original image but make sure the transferred key point still in the transferred image
    :param image:
    :param orig_key_points:
    :return:
    """
    for orig_key_point in orig_key_points:
        if orig_key_point[0] == -1:
            return None, None
    # 1. calculate maximum of zoom scale Zm
    # (1) find the maximum height and width between key points
    argmin_h = np.argmin(orig_key_points[:, 1])
    argmax_h = np.argmax(orig_key_points[:, 1])
    argmin_w = np.argmin(orig_key_points[:, 0])
    argmax_w = np.argmax(orig_key_points[:, 0])
    roi_h = orig_key_points[argmax_h][1] - orig_key_points[argmin_h][1]
    roi_w = orig_key_points[argmax_w][0] - orig_key_points[argmin_w][0]
    roi_center_h = (orig_key_points[argmax_h][1] + orig_key_points[argmin_h][1])/2
    roi_center_w = (orig_key_points[argmax_w][0] + orig_key_points[argmin_w][0])/2
    # (2) calculate maximum of zoom scale Zm
    Zm = min(image.shape[0]/roi_h, image.shape[1]/roi_w)*0.9
    # 2. generate randomly zoom scale between [1, Zm)
    zoom_scale = (Zm - 1) * random.random() + 1
    # 3. zoom image and locate the transformation center.
    # (1) enlarge image
    big_image = resize(image=image, output_shape=(int(round(image.shape[0]*zoom_scale)), int(round(image.shape[1]*zoom_scale))))
    # (2) calculate center on big_image
    center_on_big_image = [roi_center_h*zoom_scale, roi_center_w*zoom_scale]
    # (3) Clipping RoI
    LT = [int(round(center_on_big_image[0]-image.shape[0]/2)), int(round(center_on_big_image[1]-image.shape[1]/2))]
    RB = [int(round(center_on_big_image[0]+image.shape[0]/2)), int(round(center_on_big_image[1]+image.shape[1]/2))]
    if LT[0] < 0: LT[0] = 0
    if LT[1] < 0: LT[1] = 0
    if RB[0] >= big_image.shape[0]: RB[0] = big_image.shape[0]
    if RB[1] >= big_image.shape[1]: RB[1] = big_image.shape[1]
    transferred_image = big_image[LT[0]:RB[0], LT[1]:RB[1], :]
    # 4. calculate transferred key points
    #-------------------------------------------------------------------------------------------------------------------
    # (1) transformation relationship between original point and zoomed image point (which both take top left corner as
    # coordinate origin and positive direction of x axis and y axis is to the right and down, respectively)
    def orig_to_big_image(point, zoom_scale):
        return point*zoom_scale
    # (2) transformation relationship between zoomed image point (which takes RoI center as coordinate origin) and final
    # transferred image point (which takes RoI top left corner as coordinate origin). Positive direction of x axis and y
    # axis of both is to the right and down, respectively.
    def big_image_center_to_final(point, final_LT):
        return np.array([point[0] - final_LT[0], point[1] - final_LT[1]])
    #-------------------------------------------------------------------------------------------------------------------
    transferred_locations = [big_image_center_to_final(point=orig_to_big_image(point=orig_key_point[:2], zoom_scale=zoom_scale),
                                                       final_LT=(LT[1], LT[0]))
                              if orig_key_point[0] != -1 and orig_key_point[1] != -1 else [-1, -1]
                             for orig_key_point in orig_key_points]
    transferred_keypoints = [[transferred_location[0], transferred_location[1], orig_key_point[2]]
                             for transferred_location, orig_key_point in zip(transferred_locations, orig_key_points)]
    # 5. return
    # (1) if key point of transferred image out of boundary, abort transferred result and return None
    for transferred_keypoint in transferred_keypoints:
        if transferred_keypoint[1] >= transferred_image.shape[0] or transferred_keypoint[0] >= transferred_image.shape[1] \
                or transferred_keypoint[0] < 0 or transferred_keypoint[1] < 0:
            return None, None
    # (2) return transferred result otherwise
    return transferred_image, transferred_keypoints


def show_image_and_keypoints(image, keypoints, image_name=None, save_folder=None):
    """
    Show image with its all key points.
    :param image: Instance of numpy.array. (h, w, 3)
    :param keypoints: Instance of numpy.array. (keypoints_count, 2)
    :return:
    """
    plt.imshow(image)
    for key_index in range(len(keypoints)):
        plt.scatter(x=keypoints[key_index][0], y=keypoints[key_index][1], c='r')
        plt.annotate(str(key_index), (keypoints[key_index][0], keypoints[key_index][1]))
    # 1.show image and key points
    if image_name is None and save_folder is None:
        plt.show()
    # 2.save
    else:
        plt.savefig(save_folder + image_name + '.png')
    plt.close()


def write_to_excel(folder_holds_results, path_to_train_excel_file, path_to_orig_test_excel_file, save_path):
    """
    Write predicted result to csv excel file.
    :param folder_holds_results: path to folder which holds the predicted results for categories.
    Result files should in format of 'category.result'.
    :param path_to_train_excel_file:
    :param path_to_orig_test_excel_file:
    :param save_path:
    :return:
    """
    result = {}
    # 1.open train excel xlsx file
    workbook = xlrd.open_workbook(path_to_train_excel_file)
    sheet = workbook.sheet_by_index(0)
    # load original data
    orig_data = []
    for row in range(1, sheet.nrows):
        data_row = {
            'image_id': sheet.cell_value(row, 0),
            'image_category': sheet.cell_value(row, 1),
            'neckline_left': sheet.cell_value(row, 2),
            'neckline_right': sheet.cell_value(row, 3),
            'center_front': sheet.cell_value(row, 4),
            'shoulder_left': sheet.cell_value(row, 5),
            'shoulder_right': sheet.cell_value(row, 6),
            'armpit_left': sheet.cell_value(row, 7),
            'armpit_right': sheet.cell_value(row, 8),
            'waistline_left': sheet.cell_value(row, 9),
            'waistline_right': sheet.cell_value(row, 10),
            'cuff_left_in': sheet.cell_value(row, 11),
            'cuff_left_out': sheet.cell_value(row, 12),
            'cuff_right_in': sheet.cell_value(row, 13),
            'cuff_right_out': sheet.cell_value(row, 14),
            'top_hem_left': sheet.cell_value(row, 15),
            'top_hem_right': sheet.cell_value(row, 16),
            'waistband_left': sheet.cell_value(row, 17),
            'waistband_right': sheet.cell_value(row, 18),
            'hemline_left': sheet.cell_value(row, 19),
            'hemline_right': sheet.cell_value(row, 20),
            'crotch': sheet.cell_value(row, 21),
            'bottom_left_in': sheet.cell_value(row, 22),
            'bottom_left_out': sheet.cell_value(row, 23),
            'bottom_right_in': sheet.cell_value(row, 24),
            'bottom_right_out': sheet.cell_value(row, 25)
        }
        orig_data.append(data_row)
    # 2.read result files
    for root, _, files in os.walk(folder_holds_results):
        for file in files:
            if file not in [category + '.result' for category in keypoints_order.keys()]: continue
            # (1) get category
            category = os.path.splitext(file)[0]
            # (2) load file
            open_file = open(folder_holds_results + file, 'rb')
            category_pred_result = pickle.load(open_file)
            open_file.close()
            # (3) save to result
            result.update({category: category_pred_result})
    # ------------------------------------------------------------------------------------------------------------------
    # 1.new file and sheet
    rst_workbook = xlwt.Workbook()
    rst_sheet = rst_workbook.add_sheet('test', cell_overwrite_ok=True)
    key_cols_index = {
        'neckline_left': 2,
        'neckline_right': 3,
        'center_front': 4,
        'shoulder_left': 5,
        'shoulder_right': 6,
        'armpit_left': 7,
        'armpit_right': 8,
        'waistline_left': 9,
        'waistline_right': 10,
        'cuff_left_in': 11,
        'cuff_left_out': 12,
        'cuff_right_in': 13,
        'cuff_right_out': 14,
        'top_hem_left': 15,
        'top_hem_right': 16,
        'waistband_left': 17,
        'waistband_right': 18,
        'hemline_left': 19,
        'hemline_right': 20,
        'crotch': 21,
        'bottom_left_in': 22,
        'bottom_left_out': 23,
        'bottom_right_in': 24,
        'bottom_right_out': 25
    }
    # 2.write title
    rst_sheet.write(0, 0, 'image_id')
    rst_sheet.write(0, 1, 'image_category')
    for key in key_cols_index.keys():
        rst_sheet.write(0, key_cols_index[key], key)
    # 3.copy original test.csv the first two columns into new file and others -1-1-1
    # (1)open original test.csv file
    orig_workbook = xlrd.open_workbook(path_to_orig_test_excel_file)
    orig_sheet = orig_workbook.sheet_by_index(0)
    orig_record_count = orig_sheet.nrows - 1
    # (2)copy
    for row in range(1, orig_record_count + 1):
        rst_sheet.write(row, 0, orig_sheet.cell_value(row, 0))
        rst_sheet.write(row, 1, orig_sheet.cell_value(row, 1))
        for key in key_cols_index.keys():
            rst_sheet.write(row, key_cols_index[key], '-1_-1_-1')
    # ------------------------------------------------------------------------------------------------------------------
    # 1.copy predicted result
    for row in range(1, orig_record_count + 1):
        image_id = orig_sheet.cell_value(row, 0)
        image_category = orig_sheet.cell_value(row, 1)
        if image_category not in result.keys():
            print("Result for category '%s' not loaded.")
            return
        # (1) find all valid key points of current image of current category
        result_index = result[image_category]['images_id'].index(image_id)
        keypoints = result[image_category]['keypoints'][result_index]
        # (2) write valid keypoints to corresponding place
        for key_index in range(len(keypoints_order[image_category])):
            # 1> find col index
            col_index = key_cols_index[keypoints_order[image_category][key_index]]
            # 2> find x, y location
            x = int(round(keypoints[key_index][0]))
            y = int(round(keypoints[key_index][1]))
            rst_sheet.write(row, col_index, str(x) + '_' + str(y) + '_1')
    # 2.save
    rst_workbook.save(save_path)
