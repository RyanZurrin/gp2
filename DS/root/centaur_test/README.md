# centaur_test
Testing of Centaur product.

The folder structure will contain the same categories as the ones defined in Deep Health Centaur's Product Design 
Document (https://deephealth.atlassian.net/wiki/spaces/PDD/pages/60817450/DH-04+Test+List).
For instance, unit tests for "6.Input data" category will be in centaur_test/cat_6/unit_tests.py

#### Needed packages/tools:
* pytest (to run the tests)
* pytest-html (needed only for generating html reports)

#### Centaur prerequirements
Centaur repos should have been cloned to a local folder. For instance, PYTHONPATH should look something like:

/root/keras_retinanet:/root/keras:/root/centaur:/root

(Note: if you are outside the Centaur Docker container, you will need to replace "/root" with your local path)

#### Examples of use:

Assuming that you are in the centaur_test root folder, you can run the following examples: 

* `pytest hello_world.py`  <-- simplest test just to ensure that pytest was installed correctly
* `pytest hello_world.py -s`  <-- simplest test printing console output
* `pytest general_checks.py`  <-- general centaur repos and external packages check
* `pytest basic_pipeline.py`  <-- tests that run a Dxm and DBT study and compare to baseline results
* `pytest basic_pipeline.py::test_T_BP_Dxm -s`  <-- run a single test: full pipeline for a Dxm study (it takes about 80 seconds in a p3.2xlarge instance)
* `pytest basic_pipeline.py::test_T_BP_Dbt -s`  <-- run a single test: full pipeline for a Dbt study (it takes about 5 minutes in a p3.2xlarge instance)
* `pytest cat_6/unit_tests.py`  <-- run all the unit tests for category 6 (Input data)
* `pytest general_checks.py -k imports`  <-- run all the tests in 'general_checks.py' with a name pattern="*imports*"
* `pytest --html=results_preproc.html --self-contained-html`  <-- run all the tests and generate an HTML report
* `pytest --junitxml=results_preproc.xml`  <-- run all the tests and generate an XML report

#### Test Data
When a new Centaur Docker image is generated, the testing dataset is copied to /root/test_datasets folder.
Therefore, the fastest way to run the tests locally for development purposes is to create a local folder with that 
name/location and copy there the desired dataset.

For example:

```
export CENTAUR_ROOT_PATH=/home/myuser/centaur/
mkdir ${CENTAUR_ROOT_PATH}/test_datasets
gsutil cp -r gs://dh_dcm_testing/DS_01 ${CENTAUR_ROOT_PATH}/test_datasets
```

