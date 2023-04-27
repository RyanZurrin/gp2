import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_unet_collection import losses


class Util:
    @staticmethod
    def disable_tensorflow_logging():
        """ Disables tensorflow logging """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.getLogger('tensorflow').disabled = True

    @staticmethod
    def create_A_B_Z_split(images, labels, dataset_size=1000, weights=None):
        """ Creates A, B, Z split from images and labels
        Parameters
        ----------
        images : numpy array
            The images to split
        labels : numpy array
            The labels to split
        dataset_size : int
            The size of the dataset
        weights : dict
            The weights to use for the split. If None, the split is 40/40/20
        Returns
        -------
        A : numpy array
            The A split
        B : numpy array
            The B split
        Z : numpy array
            The Z split
        """

        #
        # We split as follows
        # A 0.4*dataset_size
        # B 0.4*dataset_size
        # Z 0.2*dataset_size
        #

        A_split = int(0.4 * dataset_size)
        B_split = int(0.2 * dataset_size)
        Z_split = int(0.4 * dataset_size)

        if weights:
            A_split = int(weights['A'] * dataset_size)
            B_split = int(weights['B'] * dataset_size)
            Z_split = int(weights['Z'] * dataset_size)

        # machine data
        A = images[0:A_split, :, :, 0]
        A_labels = labels[0:A_split, :, :, 0]

        A = np.stack((A, A_labels), axis=-1)

        # human data
        B = images[A_split:A_split + B_split, :, :, 0]
        B_labels = labels[A_split:A_split + B_split, :, :, 0]

        B = np.stack((B, B_labels), axis=-1)

        # we will also provide a Z array that
        # can be used to funnel additional data later

        # funnel data
        Z = images[A_split + B_split:A_split + B_split + Z_split, :, :, 0]
        Z_labels = labels[A_split + B_split:A_split + B_split + Z_split, :, :,
                   0]

        Z = np.stack((Z, Z_labels), axis=-1)

        return A, B, Z

    @staticmethod
    def create_train_val_test_split(dataset,
                                    train_count=200,
                                    val_count=300,
                                    test_count=250,
                                    shuffle=True):
        """ Creates a train, val, test split from a dataset
        Parameters
        ----------
        dataset : numpy array
            The dataset to split
        train_count : int
            The number of training samples
        val_count : int
            The number of validation samples
        test_count : int
            The number of test samples
        shuffle : bool
            Whether to shuffle the dataset before splitting
        Returns
        -------
        train : numpy array
            The training split
        val : numpy array
            The validation split
        test : numpy array
            The test split
        """

        if shuffle:
            np.random.shuffle(dataset)

        train = dataset[0:train_count]
        val = dataset[train_count:train_count + val_count]
        test = dataset[
               train_count + val_count:train_count + val_count + test_count]

        return train, val, test

    @staticmethod
    def create_numbered_file(filename, extension):
        """ Creates a numbered file
        Parameters
        ----------
        filename : str
            The filename
        extension : str
            The extension
        Returns
        -------
        numbered_file : str
            The numbered file
        """
        number = 0000
        numbered_file = filename + '_' + str(number) + extension
        while os.path.exists(numbered_file):
            number += 1
            numbered_file = filename + '_' + str(number) + extension

        return numbered_file

    @staticmethod
    def plot_accuracies(x, y1, y2):
        """ Plot line chart showing accuracies of classifier and discriminator
        Parameters
        ----------
        x : numpy.ndarray
            The x-axis values.
        y1 : numpy.ndarray
            The y-axis values for the classifier.
        y2 : numpy.ndarray
            The y-axis values for the discriminator.
        Returns
        -------
        None
        """

        # Plot Line1 (Left Y Axis)
        fig, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=80)
        line1, = ax1.plot(x, y1, color='tab:red', label='Classifier')
        line2, = ax1.plot(x, y2, color='tab:blue', label='Discriminator')
        ax1.legend(handles=[line1, line2])
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # # Plot Line2 (Right Y Axis)
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        # ax2.plot(x, y2, color='tab:blue')

        # Decorations
        # ax1 (left Y axis)
        ax1.set_xlabel('Cycle', color='tab:gray', fontsize=14)
        ax1.tick_params(axis='x', rotation=0, labelsize=12,
                        labelcolor='tab:gray')
        ax1.set_ylabel('Accuracy', color='tab:red', fontsize=14)
        ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red')
        ax1.grid(alpha=.4)

        # ax2 (right Y axis)
        # ax2.set_ylabel("# Unemployed (1000's)", color='tab:blue', fontsize=20)
        # ax2.tick_params(axis='y', labelcolor='tab:blue')
        # ax2.set_xticks(np.arange(0, len(x), 60))
        # ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
        # ax2.set_title("Personal Savings Rate vs Unemployed: Plotting in Secondary Y Axis", fontsize=22)
        fig.tight_layout()
        plt.show()

    @staticmethod
    @tf.function
    def dice_coeff(y_true, y_pred, smooth=1e-9):
        """ Calculate the dice coefficient.
        Parameters
        ----------
        y_true : numpy.ndarray
            The true masks.
        y_pred : numpy.ndarray
            The predicted masks.
        smooth : float
            The smoothing factor.
        Returns
        -------
        float
            The dice coefficient.
        """
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

    @staticmethod
    @tf.function
    def bce_dice_loss(y_true, y_pred):
        """ Calculate the loss.
    Parameters
    ----------
    y_true : numpy.ndarray
        The true masks.
    y_pred : numpy.ndarray
        The predicted masks.
    Returns
    -------
    float
        The loss.
    """
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + \
            (1 - Util.dice_coef(y_true, y_pred))

    @staticmethod
    @tf.function
    def hybrid_loss(y_true, y_pred):
        """ Calculate the loss.
        Parameters
        ----------
        y_true : numpy.ndarray
            The true masks.
        y_pred : numpy.ndarray
            The predicted masks.
        Returns
        -------
        float
            The loss.
        """
        # check that the types of both y_true and y_pred are the same
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4 / 3)
        loss_iou = losses.iou_seg(y_true, y_pred)

        # (x)
        # loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)

        return loss_focal + loss_iou  # +loss_ssim

    @staticmethod
    @tf.function
    def dice_loss(y_true, y_pred, smooth=1):
        """ Calculate the dice loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            The true masks.
        y_pred : numpy.ndarray
            The predicted masks.
        smooth : float
            The smoothing factor.

        Returns
        -------
        float
            The dice loss.
        """
        return 1 - Util.dice_coeff(y_true, y_pred, smooth)
