import os
import re
import flask
import pandas as pd

from centaur_deploy.deploys.config import Config
from deephealth_utils.data import parse_logs


class Dashboard:

    def __init__(self, output_dir):

        self.output_dir = output_dir
        logs = [f for f in os.listdir(self.output_dir) if re.search(r'.*\.log$', f)]
        assert len(logs) == 1, 'Expected exactly one log file, found the following {}'.format(logs)
        self.reader = parse_logs.LogReader(os.path.join(self.output_dir, logs[0]))
        configs = [f for f in os.listdir(self.output_dir) if re.search(r'.*_config.json$', f)]
        assert len(configs) == 1, 'Expected exactly one config file, found the following {}'.format(configs)
        self.config = Config.from_json_file(os.path.join(self.output_dir, configs[0]))
        self.collect_data()

    def collect_data(self):
        input_dir = self.config[Config.MODULE_IO, 'input_dir']
        n_input_dirs = 0
        n_input_files = 0
        for root, dirs, files in os.walk(input_dir):
            n_input_dirs += len(dirs)
            n_input_files += len(files)
        progress_df = pd.DataFrame({'Completed': [self.reader.n_files, self.reader.n_studies],
                                    'Remaining': [self.reader.n_files - n_input_files, self.reader.n_studies - n_input_dirs],
                                    'Total': [n_input_files, n_input_dirs]}, index=['File', 'Study'])
        self.tables_dict = {'Files Acceptance Criteria Counts': self.reader.count_files_ac(),
                            'Studies Acceptance Criteria Counts': self.reader.count_study_ac(),
                            'Run Progress': progress_df}

    def view_terminal(self):
        for table_name in self.tables_dict:
            print(table_name)
            print('-' * 20)
            print(self.tables_dict[table_name])
            print()

    def view_web(self):
        app = flask.Flask(__name__,
                          template_folder='templates')
        app.config['dashboard'] = self

        @app.route("/", methods=['GET'])
        def show_tables():

            tables_dict = app.config['dashboard'].tables_dict

            tables = [tables_dict[t].to_html() for t in tables_dict]
            table_names = [t for t in tables_dict]
            table_names.insert(0, '')  # for html reasons

            return flask.render_template('view.html',
                                   tables=tables,
                                   titles=table_names)

        app.run(debug=True)
