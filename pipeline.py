import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import preprocessing
from pyspark.sql import functions as F
from pyspark.sql.functions import max, mean, min, stddev, lit, regexp_replace, col

# fixes for displaying long/wide dataframes
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', -1)

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

    # fixing columns so they work with sklearn
    df['visit_start'] = pd.to_datetime(df.visit_start_date).astype('int64')
    df['visit_end'] = pd.to_datetime(df.visit_end_date).astype('int64')
    df = df.drop(columns=['visit_start_date', 'visit_end_date'])
    
    df = pd.concat([df.drop('gender_concept_name', axis=1), pd.get_dummies(df.gender_concept_name, prefix='gender')], axis=1)
    df = pd.concat([df.drop('race', axis=1), pd.get_dummies(df.race, prefix='race', drop_first=True)], axis=1)
    df = pd.concat([df.drop('ethnicity', axis=1), pd.get_dummies(df.ethnicity, prefix='ethnicity', drop_first=True)], axis=1)
    df = pd.concat([df.drop('smoking_status', axis=1), pd.get_dummies(df.smoking_status, prefix='smoking', drop_first=True)], axis=1)
    df = pd.concat([df.drop('blood_type', axis=1), pd.get_dummies(df.blood_type, prefix='blood_type', drop_first=True)], axis=1)
    df = pd.concat([df.drop('severity_type', axis=1), pd.get_dummies(df.severity_type, prefix='severity', drop_first=True)], axis=1)

    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_')
    df.columns = df.columns.str.lower()
    
    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.74e48f2e-6947-443e-ac1f-369a5575851e"),
    inpatient_encoded=Input(rid="ri.foundry.main.dataset.cef3c32e-767c-4f6a-b669-3920dac46a10")
)
def inpatient_encoded_spark(inpatient_encoded):
    return spark.createDataFrame(inpatient_encoded)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a230c6e9-ece6-46e0-89aa-c9414533899f"),
    inpatient_encoded=Input(rid="ri.foundry.main.dataset.cef3c32e-767c-4f6a-b669-3920dac46a10")
)
def pca_3_comp_analysis(inpatient_encoded):
df = inpatient_encoded
prediction = df.bad_outcome
# take out prediction column
df = df.drop(columns='bad_outcome')
scaler = preprocessing.StandardScaler()

# smaller dataframe with just a few columns for testing purposes
# sdf = df[['data_partner_id', 'age_at_visit_start_in_years_int', 'length_of_stay', 'q_score', 'testcount', 'positive_covid_test', 'negative_covid_test', 'suspected_covid', 'in_death_table', 'ecmo', 'aki_in_hospital', 'invasive_ventilation']]

# this is bad, but just fill all nulls with mean
filled_df = df.fillna(df.mean())

scaler.fit(filled_df)
scaled_df = scaler.transform(filled_df)

#start with all variables for PCA
my_pca = PCA(n_components=scaled_df.shape[1], random_state=42)
my_pca.fit(scaled_df)
pca_arr = my_pca.transform(scaled_df)

    np.cumsum(my_pca.explained_variance_ratio_ * 100)[2]

    # now the top 3 viewed with outcome
    pca_3 = PCA(n_components=3, random_state=42)
    pca_3.fit(scaled_df)
    pca_3_arr = pca_3.transform(scaled_df)

    fig = plt.figure(figsize = (12, 8))
    ax = plt.axes(projection="3d")

    splt = ax.scatter3D(
        pca_3_arr[:, 0],
        pca_3_arr[:, 1],
        pca_3_arr[:, 2],
        c = prediction,
        s=50,
        alpha=0.6)

    legend1 = ax.legend(*splt.legend_elements(), title='bad_outcome')
    ax.add_artist(legend1)

    ax.set_xlabel('First principal component')
    ax.set_ylabel('Second principal component')
    ax.set_zlabel('Third principal component')
    plt.title('PCA 3D scatter plot - ', round(np.cumsum(my_pca.explained_variance_ratio_ * 100)[2]), '% of variance captured', sep='')
    plt.show()

    # see https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
    return spark.createDataFrame(pd.DataFrame(pca_3.components_, columns=filled_df.columns,index = ['PC-1','PC-2', 'PC-3']))

@transform_pandas(
    Output(rid="ri.vector.main.execute.17ab7d1e-d4f3-4a5f-98a7-f63f5997c021"),
    inpatient_encoded=Input(rid="ri.foundry.main.dataset.cef3c32e-767c-4f6a-b669-3920dac46a10")
)
def pca_explained_variance(inpatient_encoded):
    df = inpatient_encoded
    prediction = df.bad_outcome
    # take out prediction column
    df = df.drop(columns='bad_outcome')
    scaler = preprocessing.StandardScaler()

    # this is bad, but just fill all nulls with mean
    filled_df = df.fillna(df.mean())

    scaler.fit(filled_df)
    scaled_df = scaler.transform(filled_df)

    #start with all variables for PCA
    my_pca = PCA(n_components=scaled_df.shape[1], random_state=42)
    my_pca.fit(scaled_df)
    pca_arr = my_pca.transform(scaled_df)

    plt.plot(np.cumsum(my_pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.show()

