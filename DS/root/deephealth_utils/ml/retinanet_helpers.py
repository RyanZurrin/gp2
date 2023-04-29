import sys

import keras
import numpy as np

if sys.version[0] == '2':  # Python 2.X
    sys.path.append('../keras-retinanet')
    import keras_retinanet.layers as layers
    from keras_retinanet.utils.anchors import anchor_targets_bbox, bbox_transform
    from keras_retinanet.models.retinanet import __build_anchors, AnchorParameters
elif sys.version[0] == '3':  # Python 3.X
    import keras_retinanet.layers as layers
    from keras_retinanet.utils.anchors import anchor_targets_bbox, bbox_transform
    from keras_retinanet.models.retinanet import __build_anchors, AnchorParameters

# most of this comes from keras-retinanet/preprocessing/generator.py
def compute_bb_targets(image_group, annotations_group, num_classes, batch_size=1, **kwargs):
    if not isinstance(image_group, list):
        image_group = [image_group]

    if not isinstance(annotations_group, list):
        annotations_group = [annotations_group]

    # get the max image shape
    max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

    # compute labels and regression targets
    labels_group = [None] * batch_size
    regression_group = [None] * batch_size

    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        # compute regression targets
        labels_group[index], annotations, anchors = anchor_targets_bbox(
            max_shape,
            annotations,
            num_classes,
            mask_shape=image.shape,
            **kwargs
        )
        regression_group[index] = bbox_transform(anchors, annotations)

        # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
        anchor_states = np.max(labels_group[index], axis=1, keepdims=True)
        regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

    labels_batch = np.zeros((batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
    regression_batch = np.zeros((batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

    # copy all labels and regression values to the batch blob
    for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
        labels_batch[index, ...] = labels
        regression_batch[index, ...] = regression

    return [regression_batch, labels_batch]


def create_model_ensemble(model_list, nms=True, score_threshold=0.00, max_detections=6, nms_threshold=0.00,
                          return_boxes=False, zero_benigns=True):
    """
    Creates a single model from a list of models loaded in using the argument 'convert=False' in the load_model function
    and returns a single model that averages each of the individual models' regressions and classifications before
    regressing, clipping, and filtering the bounding boxes.
    """
    # The different gui_demo models have identical layer names
    for model_idx, model in enumerate(model_list):
        for layer in model.layers:
            layer.name = 'model_%d_%s' % (model_idx, layer.name)

    models_features = []
    for model_idx, model in enumerate(model_list):
        models_features.append([model.get_layer(p_name).output for p_name in ['model_%d_P3' % model_idx,
                                                                              'model_%d_P4' % model_idx,
                                                                              'model_%d_P5' % model_idx,
                                                                              'model_%d_P6' % model_idx,
                                                                              'model_%d_P7' % model_idx]])
    anchor_parameters = AnchorParameters(
        sizes=[32, 64, 128, 256, 512],
        strides=[8, 16, 32, 64, 128],
        ratios=np.array([0.5, 1, 2], keras.backend.floatx()),
        scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx())
    )

    # The generated anchors are identical when using features from any of the models
    anchors = __build_anchors(anchor_parameters, models_features[0])

    models_regressions = [model.outputs[0] for model in model_list]
    models_classifications = [model.outputs[1] for model in model_list]
    avg_regression = keras.layers.Average()(models_regressions)
    avg_classification = keras.layers.Average()(models_classifications)

    if zero_benigns:
        z_weights = np.zeros((2, 2))
        z_weights[1, 1] = 1.0
        benign_filter_layer = keras.layers.TimeDistributed(keras.layers.Dense(2), weights=[z_weights, np.zeros(2)])
        avg_classification = benign_filter_layer(avg_classification)

    boxes = layers.RegressBoxes(name='boxes')([anchors, avg_regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model_list[0].inputs[0], boxes])

    if return_boxes:
        return keras.models.Model(inputs=[model.inputs[0] for model in model_list],
                                  outputs=[boxes, avg_classification, anchors], name='model_ensemble_anchors')

    detections = layers.FilterDetections(nms=nms, name='filtered_detections', score_threshold=score_threshold,
                                         max_detections=max_detections, nms_threshold=nms_threshold)([boxes, avg_classification])
    outputs = detections
    new_model = keras.models.Model(inputs=[model.inputs[0] for model in model_list], outputs=outputs, name='model_ensemble')

    return new_model

