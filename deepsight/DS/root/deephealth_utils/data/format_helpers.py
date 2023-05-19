def verbose_position_to_code(string):
    # dictionary based on http://dicom.nema.org/dicom/2013/output/chtml/part16/sect_CID_4014.html
    CODE_TO_ACRONYM = {'medio-lateral': 'ML',
                       'medio-lateral oblique': 'MLO',
                       'latero-medial': 'LM',
                       'latero-medial oblique': 'LMO',
                       'cranio-caudal': 'CC',
                       'caudo-cranial (from below)': 'FB',
                       'superolateral to inferomedial oblique': 'SIO',
                       'inferomedial to superolateral oblique': 'ISO',
                       'exaggerated cranio-caudal': 'XCC', # not used any more
                       'cranio-caudal exaggerated laterally': 'XCCL',
                       'cranio-caudal exaggerated medially': 'XCCM',
                       'tissue specimen from breast': 'SPECIMEN'
                      }
    if string in CODE_TO_ACRONYM:
        return CODE_TO_ACRONYM[string]
    elif string.lower().strip() == 'mediolateral oblique':
        return 'MLO'
    elif string.lower().strip() == 'craniocaudal':
        return 'CC'
    else:
        raise Exception('Could not find view code match for ' + string)


def contains_keyword(string, kws):
    for k in kws:
        if k in string:
            return True
    return False


def standard_study_description(study_desc, default_modality=None):
    if study_desc != study_desc:
        return 'None'
    else:
        study_desc = study_desc.lower()

    keywords = {}
    keywords['Biopsy'] = ['needle', 'bx', 'specimen']
    keywords['Other'] = ['inject', 'post procedure', 'localization', 'contrast',
                         'wire', 'marker', 'mag ', 'magnification', 'loc', 'compression',
                         'galactogram', 'courtesy', 'post biop', 'postbiop', 'ducto', 'unknown']
    keywords['Diagnostic'] = ['dx', 'diag', 'non screen', 'add view', 'additional view']
    keywords['Screening'] = ['screen', 'scr']
    keywords['Mammogram'] = ['mammo', 'tomo', 'dbt', 'mg', 'mam', 'mm']

    is_kw = {k: contains_keyword(study_desc, keywords[k]) for k in keywords}

    if 'biop' in study_desc and 'postbiop' not in study_desc.replace(' ', ''):
        is_kw['Biopsy'] = True

    # in the case where it just says something like Screening, a default modality can be used to assign the category
    if default_modality is not None:
        if 'screen' in study_desc:
            assert default_modality in ['Mammogram', 'Tomosynthesis'], 'Unrecognized modality: ' + default_modality
            return 'Screening' + ' ' + default_modality

    if keywords['Mammogram']:
        mamm_mod = 'Tomosynthesis' if ('3d' in study_desc or 'tomo' in study_desc or 'dbt' in study_desc) else 'Mammogram'

        if is_kw['Biopsy']:
            mamm_type = 'Biopsy'
        elif is_kw['Other']:
            mamm_type = 'Other'
        elif is_kw['Diagnostic']:
            mamm_type = 'Diagnostic'
        elif is_kw['Screening'] or study_desc.replace(' ', '') == 'mammobilateral':
            mamm_type = 'Screening'
        else:
            mamm_type = 'Diagnostic'

        return mamm_type + ' ' + mamm_mod
    elif is_kw['Biopsy']:
        return 'Biopsy'
    else:
        return 'Other'


def get_standard_study_description_mapping(study_df):
    return get_column_mapping(study_df, 'StudyDescription', 'StudyDescriptionCode')


def get_column_mapping(df, col1, col2):
    assert col1 in df, "ERROR: missing column: " + col1
    assert col2 in df, "ERROR: missing column: " + col2

    mapping = {}
    for sd, sdc in zip(df[col1], df[col2]):
        if sdc not in mapping:
            mapping[sdc] = []
        mapping[sdc].append(sd)
    for sdc in mapping:
        mapping[sdc] = list(set(mapping[sdc]))
    return mapping


def add_study_description_code(study_df):
    study_descr_code_list = []
    for study_descr in study_df['StudyDescription']:
        study_descr_code_list.append(standard_study_description(study_descr))
    study_df['StudyDescriptionCode'] = study_descr_code_list
    return study_df


def add_manufacturer_code(df):
    code_list = []
    for man in df['Manufacturer']:
        mapped_man = map_manufacturer(man)
        if mapped_man == man:  #means code couldn't be found
            mapped_man = 'UNKNOWN'
        code_list.append(mapped_man)
    df['ManufacturerCode'] = code_list
    return df


def map_manufacturer(manufacturer):
    if 'hologic' in manufacturer.lower() or 'lorad' in manufacturer.lower():
        return 'hologic'
    elif 'gemedicalsystems' in manufacturer.replace(' ', '').lower():
        return 'ge'
    else:
        return manufacturer


def map_manufacturer_model_for_specs(model_name):  #, manufacturer):
    model_name = model_name.replace(' ', '')
    if 'selenia' in model_name.lower():
        model_name = 'SeleniaDimensions'
    elif 'senograph' in model_name.lower():
        model_name = 'SenographeEssential'
    elif 'volumepreview' in model_name.lower():
        model_name = 'VolumePreview'

    return model_name


def format_dicom_tag_str(tag):
    for t in [' ', "'s", "'", '(', ')', '&', '-']:
        tag = str(tag).replace(t, '')
    return tag