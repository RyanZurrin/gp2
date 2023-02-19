import os
import sys
sys.path.insert(0,'../..')
import omama as O
import numpy as np
import sklearn
import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import pickle
THRESHOLD = 0.0001


class OutlierDetector:

    def __init__(self, DATAPATH='/raid/mpsych/'):
        """ Initializes the class
    """
        self.datapath = DATAPATH
        self.ALGORITHMS = [
            'AE',
            'AvgKNN',
            'VAE',
            'SOGAAL',
            'DeepSVDD',
            'AnoGAN',
            'HBOS',
            'LOF',
            'OCSVM',
            'IForest',
            'CBLOF',
            'COPOD',
            'SOS',
            'KDE',
            'Sampling',
            'PCA',
            'LMDD',
            'COF',
            'ECOD',
            'KNN',
            'MedKNN',
            'SOD',
            'INNE',
            'FB',
            'LODA',
            'SUOD'
        ]

    def run(self, DATASET, ALGORITHM, default_config=False):
        """ Runs the outlier detection algorithm on the dataset

    Parameters
    ----------
    DATASET : str
      The name of the dataset to run the algorithm on
    ALGORITHM : str
      The name of the algorithm to run
    default_config : bool
      Whether to use the default configuration or not

    Returns
    -------
    results : dict
      The results of the algorithm
    """

        DATASETS = {'A': 0.08,
                    'B': 0.13,
                    'C': 0.24,
                    'D': 0.24,
                    'ASTAR': 0.063,
                    'BSTAR': 0.050}

        CONTAMINATION = DATASETS[DATASET]

        # load data
        with open(os.path.join(self.datapath, 'dataset' + DATASET + '.pkl'),
                  'rb') as f:
            imgs = pickle.load(f)

        print('Loaded images.')

        # setup algorithm w/ best config w/ best feat w/ best norm
        with open(os.path.join(self.datapath,
                               'dataset' + DATASET + '_configs.pkl'),
                  'rb') as f:
            configs = pickle.load(f)

        CONFIG = configs[ALGORITHM]['config']

        if default_config:
            CONFIG = {'contamination': CONTAMINATION}

        print(CONFIG)

        NORM = configs[ALGORITHM]['norm']
        FEAT = configs[ALGORITHM]['feat']

        CONFIG['verbose'] = True
        CONFIG['return_decision_function'] = True
        CONFIG['accuracy_score'] = False

        print('Loaded config, norm, and feats.')

        feature_vector = O.Features.get_features(imgs, FEAT, NORM)

        print('Calculated features!')

        scores, labels, decision_function = O.OutlierDetector.detect_outliers(
            features=feature_vector,
            imgs=imgs,
            pyod_algorithm=ALGORITHM,
            display=False,
            number_bad=int(CONTAMINATION * 100),
            **CONFIG)

        print('Trained!')

        #
        # EVALUATE
        #

        # load groundtruth
        with open(os.path.join(self.datapath,
                               'dataset' + DATASET + '_labels.pkl'), 'rb') as f:
            groundtruth = pickle.load(f)

        evaluation = self.evaluate(groundtruth, labels)

        results = {
            'algorithm': ALGORITHM,
            'norm': NORM,
            'feat': FEAT,
            'dataset': DATASET,
            'scores': scores,
            'labels': labels,
            # 'decision_function': decision_function,
            'groundtruth': groundtruth,
            'evaluation': evaluation
        }

        return results

    def evaluate(self, groundtruth, pred):
        """ Evaluates the results of the outlier detection algorithm

    Parameters
    ----------
    groundtruth : list
      The groundtruth labels
    pred : list
      The predicted labels

    Returns
    -------
    evaluation : dict
      The evaluation metrics
    """

        cm = sklearn.metrics.confusion_matrix(groundtruth, pred)

        scores = {
            'groundtruth_indices': np.where(np.array(groundtruth) > 0),
            'pred_indices': np.where(np.array(pred) > 0),
            'roc_auc': sklearn.metrics.roc_auc_score(groundtruth, pred),
            'f1_score': sklearn.metrics.f1_score(groundtruth, pred),
            'acc_score': sklearn.metrics.accuracy_score(groundtruth, pred),
            'jaccard_score': sklearn.metrics.jaccard_score(groundtruth, pred),
            'precision_score': sklearn.metrics.precision_score(groundtruth, pred),
            'average_precision': sklearn.metrics.average_precision_score(
                groundtruth, pred),
            'recall_score': sklearn.metrics.recall_score(groundtruth, pred),
            'hamming_loss': sklearn.metrics.hamming_loss(groundtruth, pred),
            'log_loss': sklearn.metrics.log_loss(groundtruth, pred),

            'tn': cm[0, 0],
            'fp': cm[0, 1],
            'fn': cm[1, 0],
            'tp': cm[1, 1],
        }

        return scores

    def print_results(self, resultsfile):
        """ Prints the results of the outlier detection algorithm

    Parameters
    ----------
    resultsfile : str
      The path to the results file

    Returns
    -------
    None
    """
        with open(resultsfile, 'rb') as f:
            results = pickle.load(f)

        NO_RUNS = len(results[list(results.keys())[0]])

        for algo in results.keys():

            metrics = {}

            for method in results[algo][0]['evaluation'].keys():
                if method.find('indices') != -1:
                    continue

                metrics[method] = []

            for run in range(0, NO_RUNS):

                for m in metrics.keys():
                    cur = results[algo][run]['evaluation'][m]

                    metrics[m].append(cur)

            print(algo)
            for m in metrics.keys():
                # print the shape of the array
                if type(metrics[m][0]) == np.ndarray:
                    print('   ', m, metrics[m][0].shape)
                else:
                    print('   ', m, metrics[m])

                try:
                    print('   ', m, np.mean(metrics[m]), '+/-',
                          np.std(metrics[m]))
                except:
                    print('   ', m, metrics[m])

    def convert_norm_feature(self, norm, feat):
        """ Converts the norm and feature to full names with upper case first letter

    Parameters
    ----------
    norm : str
      The name of the norm
    feat : str
      The name of the feature

    Returns
    -------
    norm, feat : str
      The full names of the norm and feature
    """
        if norm == 'max':
            norm = 'Max'
        elif norm == 'minmax':
            norm = 'Min-Max'
        elif norm == 'gaussian':
            norm = 'Gaussian'

        if feat == 'hist':
            feat = 'Histogram'
        elif feat == 'downsample':
            feat = 'Downsample'
        elif feat == 'orb':
            feat = 'ORB'
        elif feat == 'sift':
            feat = 'SIFT'

        return norm, feat

    def extract_data(self, results_dict, variable='jacard_score'):
        """ Extracts the data from the results dictionary for use in tables and plots

    Parameters
    ----------
    results_dict : dict
      The results dictionary
    variable : str
      The variable to extract

    Returns
    -------
    data : dict
      The data dictionary
    """
        values = {}
        NO_RUNS = len(results_dict[list(results_dict.keys())[0]])
        for algo in results_dict.keys():
            norm = results_dict[algo][0]['norm']
            feat = results_dict[algo][0]['feat']
            norm, feat = self.convert_norm_feature(norm, feat)
            dataset = results_dict[algo][0]['dataset']
            scores = []
            for run in range(0, NO_RUNS):
                scores.append(
                    results_dict[algo][run]['evaluation'][variable])

            # operands could not be broadcast together with shapes (2,) (3,), account for this case when taking the mean
            if len(scores) == 2:
                scores = np.mean(scores, axis=0)
            else:
                scores = np.mean(scores)

            mean = np.mean(scores)

            std = np.std(scores)

            values[algo] = {
                'mean': mean,
                'std': std,
                'norm': norm,
                'feat': feat,
                'dataset': dataset
            }

        values = sorted(values.items(),
                        key=lambda x: x[1]['mean'],
                        reverse=True)
        sorted_values = dict(values)

        return sorted_values

    def result_to_latex(self,
                        resultsfile,
                        variable='jaccard_score',
                        display=True):
        """ Converts the results from a single run to a latex table

        Parameters
        ----------
        resultsfile : str
          The path to the results file
        variable : str
          The variable to extract
        display : bool
          Whether to display the table or not

        Returns
        -------
        latex : str
          The latex table
        """
        with open(resultsfile, 'rb') as f:
            results = pickle.load(f)

        vals = self.extract_data(results, variable)
        print(type(vals))
        dataset = vals[list(vals.keys())[0]]['dataset']

        latex = '''
        \\begin{table}[!h]
        \\centering
        \\label{tab:results}
        \\caption{Results for dataset %s}
        \\resizebox{\\textwidth}{!}{
        \\begin{tabular}{ccl}
        \\toprule
        \\textbf{Algorithm} & \\textbf{Norm. + Feature}  & \\textbf{%s}  \\\\
        \\midrule
        ''' % (dataset, variable.replace('_', ' ').title())
        for algo in vals.keys():
            # check if the std is not 0 or if it is below THRESHOLD
            if vals[algo]['std'] < THRESHOLD:
                latex += '%s & %s + %s & %.4f \\\\' % (
                    algo, vals[algo]['norm'], vals[algo]['feat'],
                    vals[algo]['mean']) + '\n'
            else:
                latex += '%s & %s + %s & $%.4f\pm%.4f$ \\\\' % (
                    algo, vals[algo]['norm'], vals[algo]['feat'],
                    vals[algo]['mean'],
                    vals[algo]['std']) + '\n'
        latex += '''
        \\bottomrule
        \end{tabular}
        }
        \end{table}
        '''
        if display:
            print(latex)
        return latex

    def results_to_latex(self,
                         dataset_paths: list,
                         variable='jaccard_score',
                         display=True) -> str:
        """ Prints the results from a list of datasets in a latex table format
        similar to the above method, execpt changes based on the number of datasets

        Parameters
        ----------
        dataset_paths : list
            List of paths to the results files
        variable : str
            The variable to print in the table
        display : bool
            If True, the latex table is printed to the console

        Returns
        -------
        latex : str
            The latex table
        """
        results = []
        for dataset in dataset_paths:
            with open(dataset, 'rb') as f:
                results.append(pickle.load(f))

        # extract the data from the results
        values = []
        for result in results:
            values.append(self.extract_data(result, variable))

        # get the dataset names
        dataset_names = []
        for value in values:
            dataset_names.append(value[list(value.keys())[0]]['dataset'])

        tabular = ''
        for i in range(len(dataset_names)):
            tabular += 'ccl|'
        tabular = tabular[:-1]

        multicolumn = ''
        for i in range(len(dataset_names)):
            multicolumn += '\\multicolumn{3}{c}{\\textbf{Dataset %s}} & ' % \
                           dataset_names[i]
        multicolumn = multicolumn[:-2] + '\\\\'

        header = ''
        for i in range(len(dataset_names)):
            header += '\\textbf{Algorithm} & \\textbf{Norm. + Feature}  & \\textbf{%s} &' % variable.replace(
                '_', ' ').title()
        header = header[:-1] + '\\\\'

        # create the latex table
        latex = '''
        \\begin{table}[!h]
        \\centering
        \\label{tab:results}
        \\caption{ }
        \\resizebox{\\textwidth}{!}{
        \\begin{tabular}{%s}
        \\toprule
        %s
        %s
        \\midrule
        ''' % (tabular, multicolumn, header)
        for i in range(0, len(values[0].keys())):
            for j in range(len(values)):
                algo = list(values[j].keys())[i]
                if values[j][algo]['std'] < THRESHOLD:
                    latex += '%s & %s + %s & %.4f &' % (
                        algo, values[j][algo]['norm'], values[j][algo]['feat'],
                        values[j][algo]['mean'])
                else:
                    latex += '%s & %s + %s & $%.4f\pm%.4f$ &' % (
                        algo, values[j][algo]['norm'], values[j][algo]['feat'],
                        values[j][algo]['mean'], values[j][algo]['std'])
            latex = latex[:-1] + '\\\\' + '\n'

        latex += '''
        \\bottomrule
        \end{tabular}
        }
        \end{table}
        '''
        if display:
            print(latex)
        return latex
