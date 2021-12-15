import numpy as np
import pandas as pd
import time
import json

from pathlib import Path
from scipy.io.arff import loadarff
from pandas.api import types as ptypes


def ingest_data():
    start_time = time.perf_counter()
    dataset_parser = {
        'autism':           autism_parser,
        'breast_original':  breast_parser,
        'cars':             cars_parser,
        'credit_australia': credit_australia_parser,
        'heart':            heart_parser,
        'ionosphere':       ionosphere_parser,
        'kidney':           kidney_parser,
        'krvskp':           chess_parser,
        'voting':           voting_parser,
        'airline':          airline_parser,
        'census':           census_parser,
        'kdd99':            kdd99_parser,
        'eeg':              eeg_parser,
        'credit_taiwan':    credit_taiwan_parser,
        'credit_german':    credit_german_parser,
        'compas':           compas_parser,
        'adult':            adult_parser,
    }
    base = Path(__file__).parent / 'ingestion'
    for k, v in dataset_parser.items():
        if Path(base / k).is_dir():
            try:
                v()
            except FileNotFoundError as e:
                print('data file not found. make sure you have extracted files from archive.')
                raise e
        else:
            continue
    end_time = time.perf_counter()
    print('data ingestion complete. time elapsed {} seconds.'.format(round(end_time - start_time)))


def check_dtypes(schema, df):
    # categorical -> string type or numeric
    for col in schema['categorical_columns']:
        try:
            assert ptypes.is_string_dtype(df[col]) or ptypes.is_numeric_dtype(df[col])
        except AssertionError:
            raise ValueError('Expected string or numeric but received {} for column {}'.format(df[col].dtype, col))
    # numeric -> only numeric
    for col in schema['numerical_columns']:
        try:
            assert ptypes.is_numeric_dtype(df[col])
        except AssertionError:
            raise ValueError('Expected numeric but received {} for column {}'.format(df[col].dtype, col))
    # check label (only supports binary classification)
    # try:
    #     assert df[schema['label_column']].nunique() == 2 and set(df[schema['label_column']].unique()) == {0, 1}
    # except AssertionError:
    #     raise ValueError('Expected label to be {{0, 1}} but received {}'
    #                      .format(set(df[schema['label_column']].unique())))
    # check all columns except label columns are either in numeric or categorical
    df_cols = set(df.columns) - {schema['label_column']} - {schema['id_col']}
    diff_cols = df_cols - set(schema['categorical_columns']) - set(schema['numerical_columns'])
    try:
        assert diff_cols == set()
    except AssertionError:
        raise ValueError('Expected all columns to be in either categorical or numerical columns.'
                         ' But {} is/are not.'.format(diff_cols))


def autism_parser():
    raw = loadarff(Path(__file__).parent / 'ingestion' / 'autism' / 'Autism-Adult-Data.arff')
    df = pd.DataFrame(raw[0])  # char columns are turned into binary
    binary_columns = df.select_dtypes('object').columns
    for c in binary_columns:
        df.loc[:, c] = df.loc[:, c].str.decode('utf-8')
    df.columns = [x.lower() for x in df.columns]  # lower case column names
    # missing data
    df.age.fillna(df.age.mode()[0], inplace=True)
    # label normalization
    df.rename({'class/asd': 'label'}, axis=1, inplace=True)
    df['label'].replace({'YES': 1, 'NO': 0}, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['a1_score', 'a2_score', 'a3_score', 'a4_score', 'a5_score',
                                'a6_score', 'a7_score', 'a8_score', 'a9_score', 'a10_score',
                                'gender', 'ethnicity', 'jaundice', 'autism', 'country_of_res',
                                'used_app_before', 'age_desc', 'relation'],
        'numerical_columns': ['age', 'result'],
        'label_column': 'label',
        'notes': 'labels=YES=1/NO=0. Deleted quotation characters from @attributes, '
                 'otherwise scipy.loadarff throws an error.'
                 'Missing data: age -> mode (1 instance), '
                 'ethnicity and country_of_res (95 instances each)  -> left as is i.e. \'?\''
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'autism')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'autism.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: autism written to {}'.format(out_dir / 'autism.csv'))


def breast_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'breast_original' / 'breast-cancer-wisconsin.data',
                     names=['sample_code_number', 'clump_thickness', 'cell_size_uniformity',
                            'cell_shape_uniformity', 'marginal_adhesion',
                            'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin',
                            'normal_nucleoli', 'mitoses', 'class'])
    # label normalization
    df.rename({'class': 'label'}, axis=1, inplace=True)
    df['label'].replace({2: 0, 4: 1}, inplace=True)
    # schema
    schema = {
        'id_col': 'sample_code_number',
        'categorical_columns': ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
                                'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
                                'bland_chromatin', 'normal_nucleoli', 'mitoses'],
        'numerical_columns': [],
        'label_column': 'label',
        'notes': 'labels=2=0/4=1 (2=benign, 4=malignant)'
                 'Missing data: bare_nuclei (16 instances), left as is, i.e. \'?\''
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'breast')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'breast.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: breast written to {}'.format(out_dir / 'breast.csv'))


def cars_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'cars' / 'car.data',
                     names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'evaluation'])
    # label normalization
    df.rename({'evaluation': 'label'}, axis=1, inplace=True)
    df['label'].replace({'unacc': 0, 'acc': 1, 'good': 1, 'vgood': 1}, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
        'numerical_columns': [],
        'label_column': 'label',
        'notes': 'labels= unacc=0, acc, good, vgood=1'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'cars')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'cars.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: cars written to {}'.format(out_dir / 'cars.csv'))


def credit_australia_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'credit_australia' / 'australian.dat',
                     names=['a{}'.format(n) for n in range(1, 16)],
                     sep=' ')
    # label normalization
    df.rename({'a15': 'label'}, axis=1, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['a1', 'a4', 'a5', 'a6', 'a8', 'a9', 'a11', 'a12'],
        'numerical_columns': ['a2', 'a3', 'a7', 'a10', 'a13', 'a14'],
        'label_column': 'label',
        'notes': 'label column contains 0 and 1, as opposed to 1/2 or +/- in the documentation.'
                 'The value counts are 0: 383, 1: 307, so we assume 1 to be + and 0 to be -, according'
                 'to the value counts given in the documentation.'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'credit_australia')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'credit_australia.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: credit written to {}'.format(out_dir / 'credit_australia.csv'))


def heart_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'heart' / 'heart.dat',
                     names=['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholestoral',
                            'fasting_blood_sugar', 'resting_electrocardiographic_results', 'maximum_heart_rate',
                            'exercise_induced_angina', 'oldpeak', 'slope_peak',
                            'number_of_major_vessels', 'thal', 'label'],
                     sep=' ')
    # label normalization
    df['label'].replace({1: 0, 2: 1}, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['sex', 'chest_pain_type', 'fasting_blood_sugar',
                                'resting_electrocardiographic_results',
                                'exercise_induced_angina', 'slope_peak', 'number_of_major_vessels', 'thal'],
        'numerical_columns': ['age', 'resting_blood_pressure', 'serum_cholestoral',
                              'maximum_heart_rate', 'oldpeak'],
        'label_column': 'label',
        'notes': 'labels: 1=absence=0 / 2=presence=1'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'heart')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'heart.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: heart written to {}'.format(out_dir / 'heart.csv'))


def ionosphere_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'ionosphere' / 'ionosphere.data',
                     names=['a{}'.format(n) for n in range(1, 36)])
    # label normalization
    df.rename({'a35': 'label'}, axis=1, inplace=True)
    df['label'].replace({'g': 1, 'b': 0}, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': [],
        'numerical_columns': ['a{}'.format(n) for n in range(1, 35)],
        'label_column': 'label',
        'notes': 'labels=good=1 / bad=0'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'ionosphere')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'ionosphere.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: ionosphere written to {}'.format(out_dir / 'ionosphere.csv'))


def kidney_parser():
    raw = loadarff(Path(__file__).parent / 'ingestion' / 'kidney' /
                   'Chronic_Kidney_Disease' / 'chronic_kidney_disease_full.arff')
    df = pd.DataFrame(raw[0])
    # character columns are turned into binary
    binary_columns = df.select_dtypes('object').columns
    for c in binary_columns:
        df.loc[:, c] = df.loc[:, c].str.decode('utf-8')
    # missing data
    for c in df.select_dtypes(include=np.number).isnull().any().index:
        df[c].fillna(df[c].mode()[0], inplace=True)
    # label normalization
    df.rename({'class': 'label'}, axis=1, inplace=True)
    df['label'].replace({'notckd': 0, 'ckd': 1}, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'],
        'numerical_columns': ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc'],
        'label_column': 'label',
        'notes': 'labels= ckd=1/notckd=0. Error at line ~300, there is a double comma making this line 26 elements and'
                 'corrupting the csv format. Deleted this double comma.'
                 'Missing data: numeric columns -> replaced with the mode of the column.'
                 'categorical columns -> left as is i.e. \'?\''
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'kidney')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'kidney.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: kidney written to {}'.format(out_dir / 'kidney.csv'))


def chess_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'krvskp' / 'kr-vs-kp.data',
                     names=['a{}'.format(n) for n in range(1, 38)])
    # label normalization
    df.rename({'a37': 'label'}, axis=1, inplace=True)
    df['label'].replace({'nowin': 0, 'won': 1}, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['a{}'.format(n) for n in range(1, 37)],
        'numerical_columns': [],
        'label_column': 'label',
        'notes': 'labels: nowin=0 / won=1. missing values: none'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'krvskp')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'krvskp.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: krvskp written to {}'.format(out_dir / 'krvskp.csv'))


def voting_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'voting' / 'house-votes-84.data',
                     names=['label',
                            'handicapped_infants', 'water_project_cost_sharing',
                            'budget_resolution', 'physician_fee_freeze',
                            'el_salvador_aid', 'religious_groups_in_schools',
                            'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
                            'mx_missile', 'immigration',
                            'synfuels_corporation_cutback', 'education_spending',
                            'superfund_right_to_sue', 'crime',
                            'duty_free_exports', 'export_administration_act_south_africa'])
    # label normalization
    df['label'].replace({'republican': 0, 'democrat': 1}, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['handicapped_infants', 'water_project_cost_sharing',
                                'budget_resolution', 'physician_fee_freeze',
                                'el_salvador_aid', 'religious_groups_in_schools',
                                'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
                                'mx_missile', 'immigration',
                                'synfuels_corporation_cutback', 'education_spending',
                                'superfund_right_to_sue', 'crime',
                                'duty_free_exports', 'export_administration_act_south_africa'],
        'numerical_columns': [],
        'label_column': 'label',
        'notes': 'labels: republican=0 / democrat=1. missing values -> left as is i.e. ?.'
                 'According to the documentation there are cases where the vote is neither yay nor nay.'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'voting')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'voting.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: voting written to {}'.format(out_dir / 'voting.csv'))


def airline_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'airline' / 'phpvcoG8S.csv',
                     names=['airline', 'flight', 'airport_from', 'airport_to', 'day_of_week',
                            'time', 'length', 'delay'], header=1)
    # label normalization
    df.rename({'delay': 'label'}, axis=1, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['airline', 'airport_from', 'airport_to', 'day_of_week'],
        'numerical_columns': ['flight', 'time', 'length'],
        'label_column': 'label'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'airline')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'airline.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: airline written to {}'.format(out_dir / 'airline.csv'))


def census_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'census' / 'census-income.combined',
                     names=['aage', 'aaclswkr', 'adtind', 'adtocc', 'ahga', 'ahrspay', 'ahscol',
                            'amaritl', 'amjind', 'amjocc', 'arace', 'areorgn', 'asex', 'aunmem',
                            'auntype', 'awkstat', 'capgain', 'caploss', 'divval',  'filestat',
                            'grinreg', 'grinst', 'hhdfmx', 'hhdrel', 'marsupwt', 'migmtr1', 'migmtr3',
                            'migmtr4', 'migsame', 'migsun', 'noemp', 'parent',
                            'perfntvty', 'pemntvty', 'penatvty', 'pricitshp', 'seotr',
                            'vetqva', 'vetyn', 'wkswork', 'year', 'label'])
    # label normalization
    df.drop('marsupwt', axis=1, inplace=True)
    df['label'].replace({' - 50000.': 0, ' 50000+.': 1}, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['aaclswkr', 'adtind', 'adtocc', 'ahga', 'ahscol', 'amaritl', 'amjind', 'amjocc',
                                'arace', 'areorgn', 'asex', 'aunmem', 'auntype', 'awkstat', 'filestat',
                                'grinreg', 'grinst', 'hhdfmx', 'hhdrel', 'migmtr1', 'migmtr3', 'migmtr4', 'migsame',
                                'migsun',
                                'parent', 'perfntvty', 'pemntvty', 'penatvty', 'pricitshp',
                                'seotr',  'vetqva', 'year', 'vetyn'],
             'numerical_columns': ['aage', 'ahrspay', 'capgain', 'caploss', 'divval',
                                   # 'marsupwt',
                                   'noemp', 'wkswork'],
        'label_column': 'label'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'census')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'census.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: census written to {}'.format(out_dir / 'census.csv'))


def kdd99_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'kdd99' / 'KDDCup99.csv',
                     names=['duration','protocol_type','service','flag','src_bytes','dst_bytes',
                            'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
                            'lnum_compromised','lroot_shell','lsu_attempted','lnum_root',
                            'lnum_file_creations','lnum_shells','lnum_access_files',
                            'lnum_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
                            'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
                            'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
                            'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
                            'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
                            'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
                            'dst_host_srv_rerror_rate','label'],
                            header=1)

    # label normalization
    # df['label'].replace({
    #     'back': 0,
    #     'buffer_overflow': 1,
    #     'ftp_write': 2,
    #     'guess_passwd': 3,
    #     'imap': 4,
    #     'ipsweep': 5,
    #     'land': 6,
    #     'loadmodule': 7,
    #     'multihop': 8,
    #     'neptune': 9,
    #     'nmap': 10,
    #     'normal': 11,
    #     'perl': 12,
    #     'phf': 13,
    #     'pod': 14,
    #     'portsweep': 15,
    #     'rootkit': 16,
    #     'satan': 17,
    #     'smurf': 18,
    #     'spy': 19,
    #     'teardrop': 20,
    #     'warezclient': 21,
    #     'warezmaster': 22
    # }, inplace=True)
    df['label'].replace({
        'back': 3,
        'buffer_overflow': 3,
        'ftp_write': 3,
        'guess_passwd': 3,
        'imap': 3,
        'ipsweep': 3,
        'land': 3,
        'loadmodule': 3,
        'multihop': 3,
        'neptune': 2,
        'nmap': 3,
        'normal': 0,
        'perl': 3,
        'phf': 3,
        'pod': 3,
        'portsweep': 3,
        'rootkit': 3,
        'satan': 3,
        'smurf': 1,
        'spy': 3,
        'teardrop': 3,
        'warezclient': 3,
        'warezmaster': 3
    }, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['protocol_type', 'service', 'flag', 'land',
                                'logged_in', 'is_host_login', 'is_guest_login'],
        'numerical_columns': ['duration', 'src_bytes', 'dst_bytes',
                              'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'lnum_compromised', 'lroot_shell',
                              'lsu_attempted','lnum_root','lnum_file_creations','lnum_shells','lnum_access_files','lnum_outbound_cmds',
                              'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                              'srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
                              'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
                              'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
                              'dst_host_rerror_rate','dst_host_srv_rerror_rate',],
        'label_column': 'label'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'kdd99')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'kdd99.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: kdd99 written to {}'.format(out_dir / 'kdd99.csv'))


def eeg_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'eeg' / 'phplE7q6h.csv',
                     names=['V1','V2','V3','V4','V5','V6',
                            'V7','V8','V9','V10','V11','V12',
                            'V13', 'V14', 'Class'],
                     header=1)
    # label normalization
    df.rename({'Class': 'label'}, axis=1, inplace=True)
    df['label'].replace({1: 0, 2: 1}, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': [],
        'numerical_columns': ['V1','V2','V3','V4','V5','V6','V7',
                              'V8','V9','V10','V11','V12','V13', 'V14'],
        'label_column': 'label'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'eeg')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'eeg.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: eeg written to {}'.format(out_dir / 'eeg.csv'))


def credit_taiwan_parser():
    df = pd.read_excel(Path(__file__).parent / 'ingestion' / 'credit_taiwan' / 'default of credit card clients.xls',
                       header=1)
    # label normalization
    df.rename({'default payment next month': 'label'}, axis=1, inplace=True)
    schema = {
        'id_col': 'ID',
        'categorical_columns': ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
                                'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',],
        'numerical_columns': ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                              'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                              'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',],
        'label_column': 'label'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'credit_taiwan')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'credit_taiwan.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: credit_taiwan written to {}'.format(out_dir / 'credit_taiwan.csv'))


def adult_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'adult' / 'adult.csv',
                     names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                            'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'capital-gain', 'capital-loss', 'hours-per-week',
                            'native-country', 'class'], header=0)
    # delete useless columns
    del df['fnlwgt']
    del df['education-num']
    # impute missing values
    for c in df.columns:
        if '?' in list(df[c].unique()):
            df[c].replace({'?': df[c].mode()[0]}, inplace=True)

    # label normalization
    df.rename({'class': 'label'}, axis=1, inplace=True)
    df['label'].replace({'<=50K': 0, '>50K': 1}, inplace=True)
    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['workclass', 'education',
                                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                #'capital-gain', 'capital-loss', 'hours-per-week',
                                'native-country',],
        'numerical_columns': ['age', #'fnlwgt', 'education-num',
                              #'marital-status', 'occupation', 'relationship', 'race', 'sex',
                              'capital-gain', 'capital-loss', 'hours-per-week',
                              ],
        'label_column': 'label'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'adult')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'adult.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: adult written to {}'.format(out_dir / 'adult.csv'))


def compas_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'compas' / 'compas-scores-two-years.csv')

    columns = ['age', 'age_cat', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']
    df = df[columns]

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])

    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])
    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['length_of_stay'] = df['length_of_stay'].astype(int)

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    def get_class(x):
        if x < 7:
            return 'Medium-Low'
        else:
            return 'High'
    df['class'] = df['decile_score'].apply(get_class)

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    # label normalization
    df.rename({'class': 'label'}, axis=1, inplace=True)
    df['label'].replace({'Medium-Low': 0, 'High': 1}, inplace=True)

    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['age_cat', 'sex', 'race',
                                'c_charge_degree', 'is_recid', 'is_violent_recid',
                                'two_year_recid',],
        'numerical_columns': ['age',
                              'priors_count', 'days_b_screening_arrest',
                              'length_of_stay'],
        'label_column': 'label'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'compas')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'compas.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: compas written to {}'.format(out_dir / 'compas.csv'))


def credit_german_parser():
    df = pd.read_csv(Path(__file__).parent / 'ingestion' / 'credit_german' / 'german_credit.csv',
                     names=['default', 'account_check_status', 'duration_in_month', 'credit_history',
                            'purpose', 'credit_amount', 'savings', 'present_emp_since',
                            'installment_as_income_perc', 'personal_status_sex', 'other_debtors',
                            'present_res_since', 'property', 'age', 'other_installment_plans', 'housing',
                            'credits_this_bank', 'job', 'people_under_maintenance', 'telephone', 'foreign_worker'],
                     header=0)

    # label normalization
    df.rename({'default': 'label'}, axis=1, inplace=True)

    # schema
    schema = {
        'id_col': '',
        'categorical_columns': ['account_check_status', 'credit_history',
                                'purpose', 'savings', 'present_emp_since',
                                'personal_status_sex', 'other_debtors',
                                'property', 'other_installment_plans', 'housing',
                                'job', 'telephone', 'foreign_worker'],
        'numerical_columns': ['duration_in_month',
                              'credit_amount',
                              'installment_as_income_perc',
                              'present_res_since', 'age',
                              'credits_this_bank', 'people_under_maintenance', ],
        'label_column': 'label'
    }
    check_dtypes(schema, df)
    out_dir = Path(Path(__file__).parent / 'datasets' / 'credit_german')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'credit_german.csv', header=True, index=False)
    with open(out_dir / 'schema.json', 'w') as out_file:
        json.dump(schema, out_file, indent=4)
    print('dataset: credit_german written to {}'.format(out_dir / 'credit_german.csv'))


if __name__ == '__main__':
    ingest_data()
