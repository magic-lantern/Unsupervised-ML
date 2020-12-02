import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import preprocessing
from pyspark.sql import functions as F
from pyspark.sql.functions import max, mean, min, stddev, lit, regexp_replace, col

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.cef3c32e-767c-4f6a-b669-3920dac46a10"),
    inpatient_ml_dataset=Input(rid="ri.foundry.main.dataset.07927bca-b175-4775-9c55-a371af481cc1")
)
def inpatient_encoded(inpatient_ml_dataset):
    # get rid of ids and a few other columns that are duplicates of other information
    sdf = inpatient_ml_dataset
    sdf = sdf.drop('covid_status_name')
    sdf = sdf.drop('person_id')
    sdf = sdf.drop('visit_concept_id')
    sdf = sdf.drop('visit_concept_name')
    sdf = sdf.drop('visit_occurrence_id')

    df = sdf.toPandas()
    prediction = df.bad_outcome

    # fixing columns so they work with sklearn
    df['visit_start'] = pd.to_datetime(df.visit_start_date).astype('int64')
    df['visit_end'] = pd.to_datetime(df.visit_end_date).astype('int64')
    df = df.drop(columns=['bad_outcome', 'visit_start_date', 'visit_end_date'])
    
    df = pd.concat([df.drop('gender_concept_name', axis=1), pd.get_dummies(df.gender_concept_name, prefix='gender')], axis=1)
    df = pd.concat([df.drop('race', axis=1), pd.get_dummies(df.race, prefix='race')], axis=1)
    df = pd.concat([df.drop('ethnicity', axis=1), pd.get_dummies(df.ethnicity, prefix='ethnicity')], axis=1)
    df = pd.concat([df.drop('smoking_status', axis=1), pd.get_dummies(df.smoking_status, prefix='smoking')], axis=1)
    df = pd.concat([df.drop('blood_type', axis=1), pd.get_dummies(df.blood_type, prefix='blood_type')], axis=1)
    df = pd.concat([df.drop('severity_type', axis=1), pd.get_dummies(df.severity_type, prefix='severity')], axis=1)

    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_')
    df.columns = df.columns.str.lower()

    #scaler = preprocessing.StandardScaler()
    #scaler.fit(df)
    

