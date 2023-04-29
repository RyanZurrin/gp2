# mammo_report
Creates mammogram reports for Deep Health.

## Usage 
Import from create_report.py and call create_dh_report function. 

Example of required input dictionary: 
```
report_params = {
    'dh-pdfname': 'dh-report.pdf',
    'dh-name': 'Richard Park',
    'dh-pid': '0003030303',
    'dh-dob': '2/15/1918',
    'dh-exam-date': '3/15/2018',
    'dh-sid': '307',
    'dh-modality': 'DBT',

    'dh-score-left-value': '0.10',
    'dh-img-left-top': 'img/left_top_02.jpg',
    'dh-img-left-bottom': 'img/left_bottom_02.jpg',

    'dh-score-right-value': '0.49',
    'dh-img-right-top': 'img/right_top_02.jpg',
    'dh-img-right-bottom': 'img/right_bottom_02.jpg',
    'dh-version': '0.0001'
}
create_dh_report(report_params)
```
Required fields 
- **dh-pdfname:** is the filename of the output pdf
- **dh-score-left-value** and **dh-score-right-value** will be determine the risk to be low (< 0.15), medium (<.50), or high (>.50)
- **dh-name:** Patient name
- **dh-pid:** Patient ID
- **dh-dob:** date of birth
- **dh-exam-date:** exam date
- **dh-sid:** study id
- **dh-modality:** study type i.e. DBT
- **dh-img-left-top:** left breast top image
- **dh-img-left-bottom:** left breast side image
- **dh-img-right-top:** right breast top image
- **dh-img-right-bottom:** right breast side image
- **dh-version:** software version when the report is generated 

## Installation
Install python libraries located in requirements.txt and install wkhtmltopdf to local system
```
# installing local python requirements
pip install -U -r requirements.txt

# on debian linux
sudo apt-get install wkhtmltopdf

# on macosx (using homebrew)
brew install caskroom/cask/wkhtmltopdf
```

### Example using python virtual environments
```
# Commands for creating virtual environments
mkvirtualenv deephealth  // creating virtual environment
workon deephealth // activating virtual environment
setvirtualenvproject // associates directory w/ current virtual environment
pip install -U -r requirements.txt // upgrades requirements
```

