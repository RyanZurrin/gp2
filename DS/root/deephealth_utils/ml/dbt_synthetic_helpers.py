import numpy as np

from keras.engine import Model
from keras.layers import Input

from keras_retinanet.layers import FilterDetections

from deephealth_utils.ml.detection_helpers import box_coords_to_original


class DetectionNmsSynthModel():
    def __init__(self, input_processor, det_model, slice_start_prop, slice_stride, filter_params):
        self.input_processor = input_processor
        self.det_model = det_model
        self.slice_start_prop = slice_start_prop
        self.slice_stride = slice_stride
        self.score_threshold = filter_params['score_threshold']

        boxes = Input((None, 4))
        classification = Input((None, 2))
        slice_nums = Input((None, 1))
        filter_layer = FilterDetections(**filter_params)
        inputs = [boxes, classification, slice_nums]
        outputs = filter_layer(inputs)
        self.filter_model = Model(inputs=inputs, outputs=outputs)

    def create_tracked_center_synthetic(self, X, return_all=False):
        center_slice_num = int(X.shape[0] / 2.)
        synth_im, box_coords, box_scores, box_slices, all_proc_info = \
            self.create_synthetic(X, X[center_slice_num, :, :], return_all=True)
        slice_tracking_im = np.zeros((X.shape[1], X.shape[2]), dtype='uint8') + center_slice_num
        if box_coords is not None and box_slices is not None:
            for i in reversed(range(box_coords.shape[0])):
                slice_tracking_im[box_coords[i, 1]:box_coords[i, 3], box_coords[i, 0]:box_coords[i, 2]] = box_slices[i]

        if return_all:
            return synth_im, slice_tracking_im, box_coords, box_scores, box_slices, all_proc_info
        else:
            return synth_im, slice_tracking_im, all_proc_info

    def create_synthetic(self, X, base_image, return_all=False):
        n_slices = X.shape[0]
        n_start = int(np.round(self.slice_start_prop * float(n_slices)))
        n_end = int(np.round((1. - self.slice_start_prop) * float(n_slices)))

        all_boxes = None
        all_proc_info = {}
        all_input_shapes = {}
        for slice_num in range(n_start, n_end, self.slice_stride):
            inputs, proc_info = self.input_processor(X[slice_num, :, :], return_extra_info=True)
            outs = self.det_model.predict(inputs)

            # filter out low scoring boxes
            idx = outs[1][0, :, -1] > self.score_threshold
            if np.sum(idx):
                # concatenate outputs and add slice number variable
                boxes = outs[0][:, idx, :]
                scores = outs[1][:, idx, :]
                slices = slice_num * np.ones((1, scores.shape[1], 1))
                all_proc_info[slice_num] = proc_info
                all_input_shapes[slice_num] = inputs[0][0, :, :, 0].shape

                if all_boxes is None:
                    all_boxes = boxes
                    all_scores = scores
                    all_slices = slices
                else:
                    all_boxes = np.concatenate((all_boxes, boxes), axis=1)
                    all_scores = np.concatenate((all_scores, scores), axis=1)
                    all_slices = np.concatenate((all_slices, slices), axis=1)

        synth_im = np.copy(base_image)

        if all_boxes is not None:
            filter_outs = self.filter_model.predict([all_boxes, all_scores, all_slices])

            pos_labels = filter_outs[2][0] == 1 # assumes positive label is 1

            if sum(pos_labels):
                for i in range(len(filter_outs)):
                    filter_outs[i] = filter_outs[i][0, pos_labels]
                det_idx = filter_outs[1] > self.score_threshold
                if np.sum(det_idx):
                    #print np.sum(det_idx)
                    box_coords = filter_outs[0][det_idx]
                    box_scores = filter_outs[1][det_idx]
                    box_slices = filter_outs[3][det_idx].astype(int).flatten()

                    u_box_slices = np.unique(box_slices)
                    for s in u_box_slices:
                        idx = box_slices == s

                        #proc_info = {'crop_info', 'original_shape': , 'transformed_shape': }
                        this_proc_info = {}
                        this_proc_info['original_shape'] = base_image.shape
                        this_proc_info['transformed_shape'] = all_input_shapes[s]
                        for key, val in all_proc_info[s][0].items():
                            if 'trim_image_background' in key:
                                this_proc_info['crop_info'] = val

                        orig_coords = box_coords_to_original(box_coords[idx, :], this_proc_info)
                        #orig_coords = transform_box_coords(all_input_shapes[s], base_image.shape, all_proc_info[s], box_coords[idx, :])
                        box_coords[idx, :] = orig_coords

                    box_coords = box_coords.astype(int)
                    for i in reversed(range(box_coords.shape[0])):
                        synth_im[box_coords[i, 1]:box_coords[i, 3], box_coords[i, 0]:box_coords[i, 2]] = \
                            X[box_slices[i], box_coords[i, 1]:box_coords[i, 3], box_coords[i, 0]:box_coords[i, 2]]

            else:
                box_coords = None
                box_scores = None
                box_slices = None
        else:
            box_coords = None
            box_scores = None
            box_slices = None

        if return_all:
            return synth_im, box_coords, box_scores, box_slices, all_proc_info
        else:
            return synth_im, all_proc_info

