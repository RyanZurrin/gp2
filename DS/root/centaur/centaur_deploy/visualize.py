from dashboard import dashboard
import argparse


# /Users/kevinwu/deephealth/repos/checker/logs/centaur_2020-02-11-21-36_f7beb.log
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory of a run')
args = parser.parse_args()

db = dashboard.Dashboard(args.output_dir)
db.view_terminal()
# db.view_web()