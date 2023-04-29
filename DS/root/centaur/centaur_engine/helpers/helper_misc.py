import json
import logging
import os
import time
import uuid

def read_json(fn):
    with open(fn, 'r') as fp:
        output = json.load(fp)
    return output


def save_json(fn, contents):
    with open(fn, 'w') as fp:
        json.dump(contents, fp)


def compare_jsons(j1, j2):
    import pprint
    j1 = pprint.pformat(j1)
    j2 = pprint.pformat(j2)
    return j1 == j2

def create_logger(root_output_path=None, centaur_logging_level=logging.INFO, console_logging_level=logging.DEBUG,
                  continue_log=None, supress_tensorflow_warnings=False, return_path=False):
    """
    Create a Centaur logger and a Console logger to print messages in the console, not necessarily
    with the same priority
    :param root_output_path: str. Folder that will store the Centaur log file. If None, no file will be saved
    :param centaur_logging_level: logging.X. Priority level to be logged in the centaur file
    :param console_logging_level: logging.X. Priority level to be printed in the console
    :param continue_log: pre-existing log to append to the beginning of new log
    :param ignore_tensorflow_warnings: bool. Supress Tensorflow warnings
    :param return_path: bool. Return the full path to the Centaur log file that was created if needed
    :return: Centaur logger object or tuple with (logger object, path to the Centaur file)
    """

    if root_output_path is not None and not os.path.exists(root_output_path):
        os.makedirs(root_output_path)

    centaur_logger = logging.getLogger('centaur')
    centaur_logger.setLevel(centaur_logging_level)

    # Console logger (if it does not exist already)
    system_logger = logging.getLogger()
    console_logger_found = False
    for handler in system_logger.handlers:
        if handler.name == "centaur-console":
            console_logger_found = True
            break
    if not console_logger_found:
        ch = logging.StreamHandler()
        ch.name = "centaur-console"
        ch.setLevel(console_logging_level)
        system_logger.setLevel(console_logging_level)
        system_logger.addHandler(ch)

    # File logger
    if root_output_path is not None:
        uid = uuid.uuid4().hex[0:5]
        log_file = os.path.join(root_output_path, 'centaur_{}_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'), uid))

        if continue_log is not None:  # If we want to pre-pend existing log
            # import pdb; pdb.set_trace()
            with open(continue_log, 'rb') as f1:
                with open(log_file, 'wb') as f2:
                    for line in f1:
                        f2.write(line)
            f1.close()
            f2.close()

        fh = logging.FileHandler(log_file)
        head = '%(asctime)-15s @@ %(filename)s @@ %(process)d @@ %(message)s'
        formatter = logging.Formatter(head)
        fh.setFormatter(formatter)
        fh.setLevel(centaur_logging_level)
        centaur_logger.addHandler(fh)
    else:
        log_file = None

    if supress_tensorflow_warnings:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

    if return_path:
        return centaur_logger, log_file
    else:
        return centaur_logger
