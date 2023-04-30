import datetime
import shutil
import pandas as pd
import os.path as osp
import tempfile
import pytest

from centaur_deploy.deploys.studies_db import StudiesDB
from centaur_test.data_manager import DataManager

@pytest.fixture(scope="function", autouse=True)
def override_xml_for_jenkins(record_xml_attribute):
    '''
    Override the default 'classname' property in the JUnit XML description for a clearer visualization in Jenkins
    :param record_xml_attribute:
    :return:
    '''
    record_xml_attribute("classname", "2_Unit")

def test_T_160():
    """
    Test the studies database can be created, and studies can be inserted with the right default values
    """
    temp_folder = tempfile.mkdtemp()
    try:
        db = StudiesDB(db_path=temp_folder + "centaur-tmp.db")
        dm = DataManager()
        studies = dm.get_all_studies()
        root_folder = dm.data_studies_dir
        for study in studies:
            db.insert_study(osp.join(root_folder, study))
        # Make sure the records were inserted correctly
        unstarted_studies = db.get_unstarted_studies()
        assert len(unstarted_studies) == len(studies), "Expected {} studies in the database. Got {}".format(
                                                        len(studies), len(unstarted_studies))
        for i in range(len(unstarted_studies)):
            row = unstarted_studies.iloc[i]
            db_date = datetime.datetime.fromisoformat(row['timedate_received'])
            now = datetime.datetime.utcnow()
            span = now - db_date
            assert span.seconds < 20, f"Expected dates very close. Db_date:{db_date}; Now: {now}"
            assert row['input_path'] == osp.join(root_folder, studies[i])
            # TODO: check other fields
    finally:
        shutil.rmtree(temp_folder)

# if __name__ == "__main__":
#     test_T_2_U01()