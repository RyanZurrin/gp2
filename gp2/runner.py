import gp2.data as data
import gp2.gp2 as gp2

import numpy as np
import os
import tempfile


class Runner:

    def __init__(self, verbose=False,
                 workingdir=tempfile.mkdtemp(suffix='GP2'),
                 store_after_each_step=False):

        self.store_after_each_step = store_after_each_step

        self.workingdir = workingdir

        self.verbose = verbose

        print('*** GP2 ***')
        print('Working directory:', self.workingdir)

        if verbose:
            print('Verbose mode active!')

        self.M = data.Manager()

        self.dataset_size = None

        self.classifier = None
        self.classifier_scores = []

        self.discriminator = None
        self.discriminator_scores = []

    #
    # STEP 0
    #
    def setup_data(self, images, masks, dataset_size=1000, weights=None):
        """
        Will setup:
        A_: data to train/val/test the classifier
        B_: expert labels to feed directly into the discriminator
        Z_: a repository of additional data that can further train the classifier
        """
        M = self.M

        self.dataset_size = dataset_size
        self.weights = weights

        if weights:
            # quick'n'dirty validation
            assert (weights['A'] + weights['B'] + weights['Z'] == 1)
            assert (weights['A'] * weights['A_test'] * dataset_size == weights[
                'B'] * dataset_size)
            print('Weights OK!')

        A_, B_, Z_ = gp2.Util.create_A_B_Z_split(images, masks,
                                                 dataset_size=dataset_size,
                                                 weights=weights)

        A = data.Collection.from_list(A_)
        B = data.Collection.from_list(B_)
        Z = data.Collection.from_list(Z_)

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

        A_train_, A_val_, A_test_ = gp2.Util.create_train_val_test_split(A_,
                                                                         train_count=train_count,
                                                                         val_count=val_count,
                                                                         test_count=test_count,
                                                                         shuffle=False)
        A_train_ids = A_ids[0:train_count]
        A_val_ids = A_ids[train_count:train_count + val_count]
        A_test_ids = A_ids[
                     train_count + val_count:train_count + val_count + test_count]

        A_train = data.Collection.from_list(A_train_,
                                            A_train_ids)  # COLLECTION LAND
        A_val = data.Collection.from_list(A_val_, A_val_ids)
        A_test = data.Collection.from_list(A_test_, A_test_ids)

        M.register(A_train, 'A_train')
        M.register(A_val, 'A_val')
        M.register(A_test, 'A_test')

    #
    # STEP 1
    #
    def run_classifier(self, patience_counter=2):
        """
        (Re-)Train the classifier
        """
        M = self.M

        A_train = M.get('A_train')
        A_val = M.get('A_val')
        A_test = M.get('A_test')

        #
        # ACTUAL CLASSIFIER TRAINING
        #
        if not self.classifier:
            u = gp2.UNet(verbose=self.verbose, workingdir=self.workingdir)
            self.classifier = u

        X_train_, X_train_ids = A_train.to_array()
        X_train_ = X_train_[:, :, :, 0].astype(np.float32)

        y_train_, y_train_ids = A_train.to_array()
        y_train_ = y_train_[:, :, :, 1].astype(np.float32)

        X_val_, X_val_ids = A_val.to_array()
        X_val_ = X_val_[:, :, :, 0].astype(np.float32)

        y_val_, y_val_ids = A_val.to_array()
        y_val_ = y_val_[:, :, :, 1].astype(np.float32)

        u = self.classifier
        history = u.train(X_train_, y_train_, X_val_, y_val_,
                          patience_counter=patience_counter)

        X_test_, X_test_ids = A_test.to_array()
        X_test__ = X_test_[:, :, :, 0].astype(np.float32)
        y_test_ = X_test_[:, :, :, 1].astype(np.float32)

        print('Testing the classifier...')
        predictions, scores = u.predict(X_test__, y_test_)

        #
        # A_TEST PREDICTION
        #
        A_test_pred = data.Collection.from_list(predictions, X_test_ids)

        M.register(A_test_pred, 'A_test_pred')

        self.classifier_scores.append(scores)

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step1.pickle'))

    #
    # STEP 2 (gets called by 4)
    #
    def create_C_dataset(self):
        """ Create the C dataset from the classifier predictions
        """
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

        # print('B', B_.shape)
        # print('A_test_with_pred_', A_test_with_pred_.shape)

        C_size = (2 * B_.shape[0], B_.shape[1], B_.shape[2])
        C_images_ = np.zeros((C_size + (B_.shape[3],)), dtype=B_.dtype)

        C_images_[0:A_test_with_pred_.shape[0]] = A_test_with_pred_
        C_images_[A_test_with_pred_.shape[0]:] = B_

        C_labels_ = np.empty((C_size + (1,)), dtype=np.bool)
        C_labels_[0:B_.shape[0], 0, 0, 0] = 1
        C_labels_[B_.shape[0]:, 0, 0, 0] = 0

        C_ = np.concatenate((C_images_, C_labels_), axis=-1)
        #
        #
        #

        # combine the uniq ids from A_test_pred and B
        C_ids = A_test_pred_ids + B_ids

        C = data.Collection.from_list(C_, C_ids)

        M.register(C, 'C')

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step2.pickle'))

    #
    # STEP 3 (gets called by 4)
    #
    def create_C_train_val_test_split(self, train_count=300, val_count=100,
                                      test_count=100):
        """ Create the C train/val/test split
        """

        M = self.M

        C = M.get('C')

        C.shuffle()  # we need to shuffle in connection land to keep track of the ids

        C_, C_ids = C.to_array()
        C_train_, C_val_, C_test_ = gp2.Util.create_train_val_test_split(C_,
                                                                         train_count=train_count,
                                                                         val_count=val_count,
                                                                         test_count=test_count,
                                                                         shuffle=False)

        C_train_ids = C_ids[0:train_count]
        C_val_ids = C_ids[train_count:train_count + val_count]
        C_test_ids = C_ids[
                     train_count + val_count:train_count + val_count + test_count]

        C_train = data.Collection.from_list(C_train_, C_train_ids)
        C_val = data.Collection.from_list(C_val_, C_val_ids)
        C_test = data.Collection.from_list(C_test_, C_test_ids)

        M.register(C_train, 'C_train')
        M.register(C_val, 'C_val')
        M.register(C_test, 'C_test')

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step3.pickle'))

    #
    # STEP 4 (calls 2+3+5)
    #
    def run_discriminator(self, train_ratio=0.4, val_count=0.1, test_count=0.5):
        """
        Train the discriminator using C_train/C_val.
        If the discriminator was trained, this
        will just predict.
        """
        self.create_C_dataset()

        dataset_size = self.dataset_size
        weights = self.weights

        train_count = int(0.2 * 0.4 * dataset_size)
        val_count = int(0.2 * 0.1 * dataset_size)
        test_count = int(0.2 * 0.5 * dataset_size)

        if weights:
            train_count = int(weights['B'] * weights['B_train'] * dataset_size)
            val_count = int(weights['B'] * weights['B_val'] * dataset_size)
            test_count = int(weights['B'] * weights['B_test'] * dataset_size)

        self.create_C_train_val_test_split(train_count, val_count, test_count)

        M = self.M

        if not self.discriminator:
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

            cnnd = gp2.CNNDiscriminator(verbose=self.verbose,
                                        workingdir=self.workingdir)

            cnnd.train(X_train_images_, X_train_masks_, y_train_, X_val_images_,
                       X_val_masks_, y_val_)

            self.discriminator = cnnd

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step4.pickle'))

        self.predict_discriminator()

    #
    # STEP 5 (gets called by 4)
    #
    def predict_discriminator(self):
        """
        Predict using the Discriminator (internal!)
        """
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

        C_test_pred = data.Collection.from_list(predictions, C_test_ids)

        M.register(C_test_pred, 'C_test_pred')

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step5.pickle'))

    #
    # STEP 6
    #
    def find_machine_labels(self):
        """
        This finds all machine labels,
        as indicated from the Discriminator
        and create dataset D.
        Returns number of machine labels found.
        """
        M = self.M

        C_test = M.get('C_test')
        C_test_pred = M.get('C_test_pred')

        C_test_, C_test_ids = C_test.to_array()
        C_test_pred_, C_test_pred_ids = C_test_pred.to_array()

        all_machine_labels_indices = np.where(C_test_pred_ == 1)[0]

        print('Found', len(all_machine_labels_indices), 'machine labels.')

        print('Machine labels', all_machine_labels_indices)

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

        if (len(all_machine_labels_indices) == 0):
            print('No machine labels found. Skipping step 6.')
            return 0

        assert (np.all(
            D_[0] == C_test_[all_machine_labels_indices[0]]))  # quick check

        assert (D_ids[1] == C_test_ids[all_machine_labels_indices[1]])

        print('D_ids', D_ids)

        D = data.Collection.from_list(D_, D_ids)

        M.register(D, 'D')

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step6.pickle'))

        return len(all_machine_labels_indices)

    #
    # STEP 7 (calls 8)
    #
    def relabel(self, percent_to_replace=30):
        """
        Relabels a subset of Dataset D
        """

        M = self.M

        D = M.get('D')

        # check that D in not NoneType
        if D is None:
            print('D is empty. Dataset may be too complex for this method. Skipping step 7.')
            return

        D_, D_ids = D.to_array()
        D_images = D_[:, :, :, 0]
        D_masks = D_[:, :, :, 1]
        D_labels = D_[:, 0, 0, 2]

        selected_ids = list(np.random.choice(D_ids, len(D_ids) // int(
            100 / percent_to_replace), replace=False))
        # selected_ids = D_ids[:len(D_ids)//int(100/PERCENT_TO_REPLACE)] ### for debugging

        print('Replacing', len(selected_ids), 'from', len(D_ids), '!')

        # print(len(D_ids))

        D_relabeled_ = np.empty((len(selected_ids),) + D_.shape[1:],
                                dtype=D_.dtype)

        # print(D_relabeled_.shape)

        # prefill array with image and labels, then replace labels with groundtruth!!
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
            if (k in A_test.data):
                origin = 'A_test'
            elif (k in B.data):
                origin = 'B'
            else:
                print('Lost Datapoint!!', k)  # TODO - this is a problem!
                continue

            # print(origin, k)

            ### SIMULATION CASE -> just grab ground truth###
            ### OTHERWISE THIS IS THE ENTRYPOINT FOR MANUAL RE-LABELING ###
            relabeled = M.get(origin).data[k][:, :, 1]

            D_relabeled_[i, :, :, 0] = image
            D_relabeled_[i, :, :, 1] = relabeled
            D_relabeled_[i, 0, 0, 2] = label

        print('D_relabeled_', D_relabeled_.shape[0])
        print('selected_ids', selected_ids)

        D_relabeled = data.Collection.from_list(D_relabeled_, selected_ids)

        print(D_relabeled.data.keys())

        M.register(D_relabeled, 'D_relabeled')

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step7.pickle'))

        # print('update_A_train')
        self.update_A_train()

    #
    # STEP 8
    #
    def update_A_train(self, balance=True, fillup=True):
        '''
    Update A_train with selected points from D.
    Then, remove D from B and A_test (whereever it was!).
    Fill-up both B and A_test.
    '''
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
            origin = ''
            origintext = ''
            if (k in A_test.data):
                origintext = 'A_test'
                origin = A_test
            elif (k in B.data):
                origintext = 'B'
                origin = B
            # elif (k in C_test.data):
            #     origintext = 'C_test'
            #     origin = C_test
            else:
                print('Lost Datapoint!!', k)  # TODO - this is a problem!
                continue

            p = data.Point(origin.data[k])
            p.id = k

            M.remove_and_add(origin, A_train, p)
            # print('removing', p.id, 'from', origintext, 'and adding to A_train')
            removed_counter += 1

            # now fill up the origin from Z
            if fillup:
                Z_uniq_ids = list(Z.data.keys())
                Z_uniq_id = np.random.choice(Z_uniq_ids, replace=False)

                p = data.Point(Z.data[Z_uniq_id])
                p.id = Z_uniq_id

                M.remove_and_add(Z, origin, p)
                filled_counter += 1

        print('Removed:', removed_counter, 'Filled:', filled_counter)

        if self.store_after_each_step:
            M.save(os.path.join(self.workingdir, 'M_step8.pickle'))

    #
    # PLOT!
    #
    def plot(self):

        x = range(len(self.classifier_scores))
        y1 = C_accuracies = [v[1] for v in self.classifier_scores]
        y2 = D_accuracies = [v[1] for v in self.discriminator_scores]

        gp2.Util.plot_accuracies(x, y1, y2)
