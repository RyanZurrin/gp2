import json



def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), convert_keys_to_string(v))
                for k, v in dictionary.items())


def load_config(config_path):
    try:
        config = json.loads(open(config_path, 'r').read())
    except IOError:
        raise IOError("Config file cannot be read ({})".format(config_path))
    except ValueError:
        raise ValueError("Config file is not in JSON format")
    return convert_keys_to_string(config)


def configs_equal(baseline_config, config):
    """
    Compare two config objects ignoring some of the fields
    :param baseline_config: Config object
    :param config: Config object
    :return: bool. True if baseline_config and config are equal when ignoring some of the fields
    """
    ignored_tuples = (
        ("DEPLOY", "exit_on_error"),
        ("ENGINE", "preprocessed_numpy_folder"),
        ("ENGINE", "save_to_ram"),
        ("ENGINE", "reuse_ds"),
        ("ENGINE", "input_dir"),
        ("IO", "input_dir"),
        ("IO", "output_dir"),
        ("DEPLOY", "reports"),
    )

    for category in baseline_config.MODULES:
        for param_key in baseline_config[category].keys():
            if (category, param_key) in ignored_tuples:
                continue
            if baseline_config[category, param_key] != config[category, param_key]:
                print("{}-{} wrong. Baseline: {}. Computed: {}".format(category, param_key,
                                                                       baseline_config[category, param_key],
                                                                       config[category, param_key]))
                return False
    return True