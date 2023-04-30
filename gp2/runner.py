from .data import *
from .gp2 import UNet, UNetPLUS, KUNet, KATTUnet2D, KR2UNet2dD, KResUNet2D, \
    KUNet2D, KUNet3Plus2D, KUNetPlus2D, KVNet2D, CNNDiscriminator, \
    CNNDiscriminatorPLUS, Util
import time
import numpy as np
import os
import tempfile


def validate_weights(weights, tolerance=1e-6):
    """ Validate the weights for training.

    What must be verified:
        A_train + A_val + A_test = 1. \n
        B_train + B_val + B_test = 1. \n
        A + B + Z = 1. \n
        A * A_test = B. \n
        Z > .1.

    Parameters
    ----------
    weights : dict
        Weights to use for training. If None, will use the default weights.
        Weights should be a dictionary with keys containing the following:
        'A_train', 'A_val', 'A_test', 'B_train','B_val', 'B_test', 'A', 'B',
        'Z'
    tolerance : float
        Tolerance to use for the validation.

    Returns
    -------
    bool
        True if the weights are valid.

    Raises
    ------
    ValueError
        If the weights are not valid.
    """
    # Check if A_train + A_val + A_test = 1
    A_sum = weights['A_train'] + weights['A_val'] + weights['A_test']
    if not np.isclose(A_sum, 1, rtol=tolerance):
        raise ValueError("A_train + A_val + A_test must be equal to 1")

    # Check if B_train + B_val + B_test = 1
    B_sum = weights['B_train'] + weights['B_val'] + weights['B_test']
    if not np.isclose(B_sum, 1, rtol=tolerance):
        raise ValueError("B_train + B_val + B_test must be equal to 1")

    # Check if A + B + Z = 1
    main_sum = weights['A'] + weights['B'] + weights['Z']
    if not np.isclose(main_sum, 1, rtol=tolerance):
        raise ValueError("A + B + Z must be equal to 1")

    # Check if A * A_test = B
    A_mul_A_test = weights['A'] * weights['A_test']
    if not np.isclose(A_mul_A_test, weights['B'], rtol=tolerance):
        raise ValueError("A * A_test must be equal to B")

    # Check if Z > .1
    if weights['Z'] < .1:
        raise ValueError("Z must be greater than .1")

    print("Weights OK!")
    return True


class Runner:
    discriminatorTrained = False

    def __init__(self,
                 verbose=False,
                 workingdir=tempfile.mkdtemp(suffix='GP2'),
                 store_after_each_step=False,
                 classifier=None,
                 discriminator=None,
                 weights=None,
                 **kwargs):
        """ Initialize the GP2 runner with specified classifier and
        discriminator.

        Parameters
        ----------
        verbose : bool
            If True, will print out additional information during training.
        workingdir : str
            Location where to store temporary files and model checkpoints.
        store_after_each_step : bool
            If True, will store the model after each step of the training
            process.
        classifier : str or Classifier
            The classifier to use. If None or 'unet', will use the default
            handcrafted  unet classifier. Supported classifiers are:
            'unet','unetplus', 'kattunet2d', 'kunet2d', 'kunetplus2d',
            'kresunet2d', 'kunet3plus2d', 'kvnet2d', 'kr2unet2d'
        discriminator : str or Discriminator
            The discriminator to use. If None or 'cnn', will use the default
            handcrafted cnn discriminator.
        **kwargs : dict
            Additional keyword arguments to pass to the classifier and
            discriminator.
        """

        self.weights = weights
        self.store_after_each_step = store_after_each_step

        self.workingdir = workingdir

        self.verbose = verbose

        self.M = Manager()

        self.dataset_size = None

        # Initialize the classifier
        self.classifier_scores = []
        if classifier is None or isinstance(classifier,
                                            UNet) or classifier == 'unet':
            self.classifier = UNet(verbose=self.verbose,
                                   workingdir=self.workingdir)
        elif isinstance(classifier, KUNet) or classifier == 'kunet':
            self.classifier = KUNet(verbose=self.verbose,
                                    workingdir=self.workingdir,
                                    **kwargs)
        elif isinstance(classifier, UNetPLUS) or classifier == 'unetplus':
            self.classifier = UNetPLUS(verbose=self.verbose,
                                       workingdir=self.workingdir, **kwargs)
        elif isinstance(classifier, KATTUnet2D) or classifier == 'kattunet2d':
            self.classifier = KATTUnet2D(verbose=self.verbose,
                                         workingdir=self.workingdir,
                                         **kwargs)
        elif isinstance(classifier, KUNet2D) or classifier == 'kunet2d':
            self.classifier = KUNet2D(verbose=self.verbose,
                                      workingdir=self.workingdir, **kwargs)
        elif isinstance(classifier, KUNetPlus2D) or classifier == 'kunetplus2d':
            self.classifier = KUNetPlus2D(verbose=self.verbose,
                                          workingdir=self.workingdir,
                                          **kwargs)
        elif isinstance(classifier, KResUNet2D) or classifier == 'kresunet2d':
            self.classifier = KResUNet2D(verbose=self.verbose,
                                         workingdir=self.workingdir,
                                         **kwargs)
        elif isinstance(classifier,
                        KUNet3Plus2D) or classifier == 'kunet3plus2d':
            self.classifier = KUNet3Plus2D(verbose=self.verbose,
                                           workingdir=self.workingdir,
                                           **kwargs)
        elif isinstance(classifier, KVNet2D) or classifier == 'kvnet2d':
            self.classifier = KVNet2D(verbose=self.verbose,
                                      workingdir=self.workingdir, **kwargs)
        elif isinstance(classifier, KR2UNet2dD) or classifier == 'kr2unet2d':
            self.classifier = KR2UNet2dD(verbose=self.verbose,
                                         workingdir=self.workingdir,
                                         **kwargs)
        else:
            raise ValueError('Classifier not supported: {}'.format(classifier))

        # Initialize the discriminator
        self.discriminator_scores = []
        if discriminator is None or isinstance(
                discriminator, CNNDiscriminator) or discriminator == 'cnn':
            print('Using default discriminator (CNN)')
            self.discriminator = CNNDiscriminator(
                verbose=self.verbose, workingdir=self.workingdir)
        elif isinstance(discriminator,
                        CNNDiscriminatorPLUS) or discriminator == 'cnnplus':
            print('Using  discriminator (CNN+)')
            self.discriminator = CNNDiscriminatorPLUS(
                verbose=self.verbose, workingdir=self.workingdir)
        else:
            raise ValueError('Discriminator not supported: {}'.format(
                discriminator))

    #
    # STEP 0
    #
    def setup_data(self, images, masks, dataset_size=1000, weights=None):
        """ Set up the data for training.

        Each dataset is composed of three parts:
            A_: data to train/val/test the classifier \n
            B_: expert labels to feed directly into the discriminator \n
            Z_: a repository of additional data that can further train the
            classifier

        Parameters
        ----------
        images : list of np.ndarray
            List of images to use for training.
        masks : list of np.ndarray
            List of masks to use for training.
        dataset_size : int
            Number of images to use for training.
        weights : dict
            Weights to use for training. If None, will use the default weights.
            Weights should be a dictionary with keys 'A', 'B', 'Z', 'A_test'.
            The weights should sum to 1.0.

        Returns
        -------
        None
        """
        M = self.M

        self.dataset_size = dataset_size
        self.weights = weights

        if weights:
            validate_weights(weights)

        A_, B_, Z_ = Util.create_A_B_Z_split(images, masks,
                                             dataset_size=dataset_size,
                                             weights=weights)

        A = Collection.from_list(A_)
        B = Collection.from_list(B_)
        Z = Collection.from_list(Z_)

        M.register(A, 'A')  # we might not need to save this one here
        M.register(B, 'B')
        M.register(Z, 'Z')

        A = M.get('A')
        A_, A_ids = A.to_array()

        # if no weights configured, fallback
        train_count = int(0.4 * 0.1 * dataset_size)
        val_count = int(0.4 * 0.4 * dataset_size)
        test_count = int(0.4 * 0.5 * dataset_size)

        if weights:
            train_count = int(weights['A'] * weights['A_train'] * dataset_size)
            val_count = int(weights['A'] * weights['A_val'] * dataset_size)
            test_count = int(weights['A'] * weights['A_test'] * dataset_size)

        A_train_, A_val_, A_test_ = Util.create_train_val_test_split(
            A_, train_count=train_count, val_count=val_count,
            test_count=test_count, shuffle=False)
        A_train_ids = A_ids[0:train_count]
        A_val_ids = A_ids[train_count:train_count + val_count]
        A_test_ids = A_ids[
                     train_count + val_count:train_count + val_count + test_count]

        A_train = Collection.from_list(A_train_, A_train_ids)  # COLLECTION LAND
        A_val = Collection.from_list(A_val_, A_val_ids)
        A_test = Collection.from_list(A_test_, A_test_ids)

        M.register(A_train, 'A_train')
        M.register(A_val, 'A_val')
        M.register(A_test, 'A_test')

    #
    # STEP 1
    #
    def run_classifier(self, patience_counter=2, epochs=100, batch_size=64):
        """ (Re-)Train the classifier.

        Parameters
        ----------
        patience_counter : int
            Number of epochs to wait before early stopping.
        epochs : int
            Number of epochs to train for.
        batch_size : int
            Batch size to use for training.

        Returns
        -------
        None
        """
        M = self.M

        A_train = M.get('A_train')
        A_val = M.get('A_val')
        A_test = M.get('A_test')

        #
        # ACTUAL CLASSIFIER TRAINING
        #
        u = self.classifier

        X_train_, X_train_ids = A_train.to_array()
        X_train_ = X_train_[:, :, :, 0].astype(np.float32)

        y_train_, y_train_ids = A_train.to_array()
        y_train_ = y_train_[:, :, :, 1].astype(np.float32)

        X_val_, X_val_ids = A_val.to_array()
        X_val_ = X_val_[:, :, :, 0].astype(np.float32)

        y_val_, y_val_ids = A_val.to_array()
        y_val_ = y_val_[:, :, :, 1].astype(np.float32)

        u.train(X_train_, y_train_, X_val_, y_val_,
                patience_counter=patience_counter,
                epochs=epochs, batch_size=batch_size)

        X_test_, X_test_ids = A_test.to_array()
        X_test__ = X_test_[:, :, :, 0].astype(np.float32)
        y_test_ = X_test_[:, :, :, 1].astype(np.float32)

        print('Testing the classifier...')
        predictions, scores = u.predict(X_test__, y_test_)

        #
        # A_TEST PREDICTION
        #
        A_test_pred = Collection.from_list(predictions, X_test_ids)

        M.register(A_test_pred, 'A_test_pred')

        self.classifier_scores.append(scores)

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step1.pickle'))

    #
    # STEP 2 (gets called by 4)
    #
    def create_C_dataset(self):
        """ Create the C dataset from the classifier predictions (internal!)."""
        M = self.M

        A_test = M.get('A_test')
        A_test_pred = M.get('A_test_pred')
        B = M.get('B')

        A_test_images_only_, A_test_images_only_ids = A_test.to_array()
        A_test_images_only_ = A_test_images_only_[:, :, :, 0].astype(np.uint8)

        A_test_pred_, A_test_pred_ids = A_test_pred.to_array()
        A_test_pred_ = A_test_pred_.astype(np.uint8)

        A_test_with_pred_ = np.stack(
            (A_test_images_only_, A_test_pred_[:, :, :, 0]), axis=-1)

        #
        # CREATE C DATASET
        #
        B_, B_ids = B.to_array()

        C_size = (2 * B_.shape[0], B_.shape[1], B_.shape[2])
        C_images_ = np.zeros((C_size + (B_.shape[3],)), dtype=B_.dtype)

        C_images_[0:A_test_with_pred_.shape[0]] = A_test_with_pred_
        C_images_[A_test_with_pred_.shape[0]:] = B_

        C_labels_ = np.empty((C_size + (1,)), dtype=np.bool)
        C_labels_[0:B_.shape[0], 0, 0, 0] = 1
        C_labels_[B_.shape[0]:, 0, 0, 0] = 0

        C_ = np.concatenate((C_images_, C_labels_), axis=-1)

        # combine the uniq ids from A_test_pred and B
        C_ids = A_test_pred_ids + B_ids

        C = Collection.from_list(C_, C_ids)

        M.register(C, 'C')

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step2.pickle'))

    #
    # STEP 3 (gets called by 4)
    #
    def create_C_train_val_test_split(self,
                                      train_count=300,
                                      val_count=100,
                                      test_count=100):
        """ Create the C train/val/test split from the C dataset (internal!).

        Parameters
        ----------
        train_count : int
            Number of training samples.
        val_count : int
            Number of validation samples.
        test_count : int
            Number of test samples.

        Returns
        -------
        None
        """

        M = self.M

        C = M.get('C')
        # we need to shuffle in connection land to keep track of the ids
        C.shuffle()

        C_, C_ids = C.to_array()
        C_train_, C_val_, C_test_ = Util.create_train_val_test_split(
            C_, train_count=train_count, val_count=val_count,
            test_count=test_count, shuffle=False)

        C_train_ids = C_ids[0:train_count]
        C_val_ids = C_ids[train_count:train_count + val_count]
        C_test_ids = \
            C_ids[train_count + val_count:train_count + val_count + test_count]

        C_train = Collection.from_list(C_train_, C_train_ids)
        C_val = Collection.from_list(C_val_, C_val_ids)
        C_test = Collection.from_list(C_test_, C_test_ids)

        M.register(C_train, 'C_train')
        M.register(C_val, 'C_val')
        M.register(C_test, 'C_test')

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step3.pickle'))

    #
    # STEP 4 (calls 2+3+5)
    #
    def run_discriminator(self, epochs=100, batch_size=64, patience_counter=2,
                          train_ratio=0.4, val_ratio=0.1, test_ratio=0.5,
                          threshold=1e-6):
        """ Train the discriminator using C_train/C_val. If the discriminator was
        trained, this will just predict.

        Parameters
        ----------
        epochs : int
            Number of epochs.
        batch_size : int
            Batch size.
        patience_counter : int
            Patience counter.
        train_ratio : float
            Ratio of training samples.
        val_ratio : float
            Ratio of validation samples.
        test_ratio : float
            Ratio of test samples.
        threshold : float
            Threshold for the sum of train_ratio, val_ratio, and test_ratio.

        Returns
        -------
        None
        """
        # Check that the sum of the ratios is approximately equal to 1
        if not (
                1 - threshold <= train_ratio + val_ratio + test_ratio <= 1 + threshold):
            raise ValueError(
                "The sum of train_ratio, val_ratio, and test_ratio must be approximately equal to 1")

        self.create_C_dataset()

        dataset_size = self.dataset_size
        weights = self.weights

        train_count = int(0.2 * train_ratio * dataset_size)
        val_count = int(0.2 * val_ratio * dataset_size)
        test_count = int(0.2 * test_ratio * dataset_size)

        if weights:
            train_count = int(weights['B'] * weights['B_train'] * dataset_size)
            val_count = int(weights['B'] * weights['B_val'] * dataset_size)
            test_count = int(weights['B'] * weights['B_test'] * dataset_size)

        self.create_C_train_val_test_split(train_count, val_count, test_count)

        M = self.M

        if self.discriminatorTrained is False:
            print("****** TRAINING DISCRIMINATOR ******")
            C_train = M.get('C_train')
            C_val = M.get('C_val')

            C_train_, C_train_ids = C_train.to_array()
            X_train_images_ = C_train_[:, :, :, 0]
            X_train_masks_ = C_train_[:, :, :, 1]
            y_train_ = C_train_[:, 0, 0, 2]

            C_val_, C_val_ids = C_val.to_array()
            X_val_images_ = C_val_[:, :, :, 0]
            X_val_masks_ = C_val_[:, :, :, 1]
            y_val_ = C_val_[:, 0, 0, 2]

            self.discriminator.train(X_train_images_, X_train_masks_, y_train_,
                                     X_val_images_, X_val_masks_, y_val_,
                                     patience_counter=patience_counter,
                                     epochs=epochs, batch_size=batch_size)
            self.discriminatorTrained = True

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step4.pickle'))

        self.predict_discriminator()

    #
    # STEP 5 (gets called by 4)
    #
    def predict_discriminator(self):
        """ Predict using the Discriminator (internal!) """
        M = self.M

        C_test = M.get('C_test')
        C_test_, C_test_ids = C_test.to_array()
        X_test_images_ = C_test_[:, :, :, 0]
        X_test_masks_ = C_test_[:, :, :, 1]
        y_test_ = C_test_[:, 0, 0, 2]

        cnnd = self.discriminator
        print('Testing the discriminator...')
        predictions, scores = cnnd.predict(X_test_images_, X_test_masks_,
                                           y_test_)

        self.discriminator_scores.append(scores)

        C_test_pred = Collection.from_list(predictions, C_test_ids)

        M.register(C_test_pred, 'C_test_pred')

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step5.pickle'))

    #
    # STEP 6
    #
    def find_machine_labels(self):
        """ This finds all machine labels, as indicated from the Discriminator
        and create dataset D.  Returns number of machine labels found.
        """
        M = self.M

        C_test = M.get('C_test')
        C_test_pred = M.get('C_test_pred')

        C_test_, C_test_ids = C_test.to_array()
        C_test_pred_, C_test_pred_ids = C_test_pred.to_array()

        all_machine_labels_indices = np.where(C_test_pred_ == 1)[0]

        assert (C_test_ids == C_test_pred_ids)  # must be the same

        #
        # CREATE D DATASET
        #
        D_ = np.empty(((len(all_machine_labels_indices),) + C_test_.shape[1:]),
                      dtype=C_test_.dtype)
        D_ids = []

        for i, p in enumerate(all_machine_labels_indices):
            D_[i] = C_test_[p]
            D_ids.append(C_test_ids[p])

        if len(all_machine_labels_indices) == 0:
            print('No machine labels found. Skipping step 6.')
            return 0

        assert (np.all(
            D_[0] == C_test_[all_machine_labels_indices[0]]))  # quick check

        assert (D_ids[1] == C_test_ids[all_machine_labels_indices[1]])

        D = Collection.from_list(D_, D_ids)

        M.register(D, 'D')

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step6.pickle'))

        return len(all_machine_labels_indices)

    #
    # STEP 7 (calls 8)
    #
    def relabel(self, percent_to_replace=30, balance=False, fillup=True):
        """ Relabels a subset of Dataset D

        Parameters
        ----------
        percent_to_replace : int
            Percentage of D to relabel
        balance : bool
            Whether to balance when updating A_train
        fillup : bool
            Whether to fillup when updating A_train

        Returns
        -------
        None
        """

        M = self.M

        D = M.get('D')

        # check that D in not NoneType
        if D is None:
            print(
                'D is empty. Dataset may be too complex for this method. Skipping step 7.')
            return

        D_, D_ids = D.to_array()

        selected_ids = list(np.random.choice(D_ids, len(D_ids) // int(
            100 / percent_to_replace), replace=False))

        print('Replacing', len(selected_ids), 'from', len(D_ids), '!')

        D_relabeled_ = np.empty((len(selected_ids),) + D_.shape[1:],
                                dtype=D_.dtype)

        A_test = M.get('A_test')
        B = M.get('B')

        uniqids_in_D = list(D.data.keys())

        for i, k in enumerate(selected_ids):
            # i is running number 0..len(selected_ids)
            # k is the uniqid of a datapoint

            # j is the position of the uniqid in D
            j = uniqids_in_D.index(k)

            # grab image
            image = D_[j, :, :, 0]
            label = D_[j, 0, 0, 2]

            origin = ''
            if k in A_test.data:
                origin = 'A_test'
            elif k in B.data:
                origin = 'B'
            else:
                print('Lost Datapoint!!', k)
                continue

            ### SIMULATION CASE -> just grab ground truth###
            ### OTHERWISE THIS IS THE ENTRYPOINT FOR MANUAL RE-LABELING ###
            relabeled = M.get(origin).data[k][:, :, 1]

            D_relabeled_[i, :, :, 0] = image
            D_relabeled_[i, :, :, 1] = relabeled
            D_relabeled_[i, 0, 0, 2] = label

        print('D_relabeled_', D_relabeled_.shape[0])

        D_relabeled = Collection.from_list(D_relabeled_, selected_ids)

        M.register(D_relabeled, 'D_relabeled')

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step7.pickle'))

        self.update_A_train(balance=balance, fillup=fillup)

    #
    # STEP 8
    #
    def update_A_train(self, balance=False, fillup=True):
        """ Update A_train with selected points from D. Then, remove D from B
        and A_test (wherever it was!). Fill-up both B and A_test (internal!).

        Parameters
        ----------
        balance : bool
            If True, balance A_train with B and A_test
        fillup : bool
            If True, fill-up B and A_test with points from A_train

        Returns
        -------
        None
        """
        M = self.M

        D_relabeled = M.get('D_relabeled')

        A_train = M.get('A_train')
        A_test = M.get('A_test')
        B = M.get('B')

        Z = M.get('Z')  # repository

        removed_counter = 0
        filled_counter = 0

        point_ids = list(D_relabeled.data.keys())

        print('point ids', len(point_ids))

        # Move points from A_test to A_train
        for k in point_ids:
            # we need to check where this datapoint originally came from
            if k in A_test.data:
                origintext = 'A_test'
                origin = A_test
            elif k in B.data:
                origintext = 'B'
                origin = B
            else:
                print('Lost Datapoint!!', k)
                continue

            p = Point(origin.data[k])
            p.id = k

            M.remove_and_add(origin, A_train, p)
            removed_counter += 1

            # now fill up the origin from Z
            if fillup:
                Z_uniq_ids = list(Z.data.keys())
                Z_uniq_id = np.random.choice(Z_uniq_ids, replace=False)

                p = Point(Z.data[Z_uniq_id])
                p.id = Z_uniq_id

                M.remove_and_add(Z, origin, p)
                filled_counter += 1

        if balance:
            total_samples = len(A_train.data.keys()) + \
                            len(B.data.keys()) + len(A_test.data.keys())
            target_samples = total_samples // 3

            while abs(len(A_train.data) - len(B.data)) > 1 or abs(
                    len(A_train.data) - len(A_test.data)) > 1:
                if len(A_train.data) > target_samples:
                    if len(B.data) < target_samples:
                        self.transfer(A_train, B)
                    elif len(A_test.data) < target_samples:
                        self.transfer(A_train, A_test)
                elif len(B.data) > target_samples:
                    if len(A_train.data) < target_samples:
                        self.transfer(B, A_train)
                    elif len(A_test.data) < target_samples:
                        self.transfer(B, A_test)
                elif len(A_test.data) > target_samples:
                    if len(A_train.data) < target_samples:
                        self.transfer(A_test, A_train)
                    elif len(B.data) < target_samples:
                        self.transfer(A_test, B)

        print('Removed:', removed_counter, 'Filled:', filled_counter)

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step8.pickle'))

    def transfer(self, source, target):
        """ Transfer one point from source to target.

        Parameters
        ----------
        source : Collection
            Source collection
        target : Collection
            Target collection

        Returns
        -------
        None
        """
        print('Transfer from', source.name, 'to', target.name)
        M = self.M
        source_uniq_ids = list(source.data.keys())
        source_uniq_id = np.random.choice(source_uniq_ids, replace=False)

        p = Point(source.data[source_uniq_id])
        p.id = source_uniq_id

        M.remove_and_add(source, target, p)

    def run(self,
            images,
            masks,
            weights,
            runs=1,
            epochs=100,
            batch_size=64,
            patience_counter=2,
            percent_to_replace=30,
            balance=False,
            fillup=True):
        """ Run the whole GP2 algorithm, including setting up the data, running
        the classifier and discriminator, and relabeling.

        Parameters
        ----------
        images : np.ndarray
            List of images
        masks : np.ndarray
            List of masks
        weights : dict
            Dictionary of weights for the different classes
        runs : int
            Number of runs
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        patience_counter : int
            Number of times the classifier can run without improvement. (Default: 2)
        percent_to_replace : int
            Percentage of points to replace in each run. (Default: 30)
        balance : bool
            If True, balance A_train with B and A_test
        fillup : bool
            If True, fill-up B and A_test with points from A_train

        Returns
        --------
        None
        """
        # assert that len of images and masks is the same
        assert len(images) == len(masks)
        dataset_size = len(images)
        self.setup_data(images=images, masks=masks,
                        dataset_size=dataset_size,
                        weights=weights)

        for run in range(runs):
            print('******')
            print('Loop', run+1)
            t0 = time.time()
            self.run_classifier(patience_counter=patience_counter,
                                epochs=epochs, batch_size=batch_size)
            self.run_discriminator(patience_counter=patience_counter,
                                   epochs=epochs, batch_size=batch_size)
            l = self.find_machine_labels()
            if l == 0:
                print('No more machine labels.')
                print('TOOK', time.time() - t0, 'seconds')
                break
            self.relabel(percent_to_replace=percent_to_replace,
                         balance=balance,
                         fillup=fillup)
            print('TOOK', time.time() - t0, 'seconds')
            print('==== DONE LOOP', run+1, '====')

    #
    # PLOT!
    #
    def plot(self):
        """ Plot the accuracies of the classifier and discriminator based on
        the scores stored in classifier_scores and discriminator_scores.

        Returns:
        --------
        None
        """
        x = range(len(self.classifier_scores))
        y1 = [v[1] for v in self.classifier_scores]
        y2 = [v[1] for v in self.discriminator_scores]

        Util.plot_accuracies(x, y1, y2)
