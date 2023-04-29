import argparse

from centaur_deploy.deploys.studies_db import StudiesDB


class Receiver(object):

    def __init__(self, db=None):
        self.db = db

    def process_dir(self, path):
        """ scan directory for images """
        try:
            if path:
                # remove trailing slash
                path = path.rstrip('/')
                self.db.insert_study(path)
        except Exception as e:
            # import pdb
            # pdb.set_trace()
            raise Exception(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process new studies from PACS')
    parser.add_argument("--directory", "-d", type=str, required=True, help="directory to scan for changes")

    args = parser.parse_args()
    directory = args.directory
    db = StudiesDB()
    receiver = Receiver(db=db)
    print("Receiving study {}...".format(directory))
    # try:
    # process the directory provided in the command line
    receiver.process_dir(directory)
    print("Study {} processed".format(directory))
    # except Exception as e:
    #     raise Exception(e)
