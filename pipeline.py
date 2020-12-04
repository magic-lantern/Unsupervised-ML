import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from pyspark.sql import functions as F
from pyspark.sql.functions import max, mean, min, stddev, lit, regexp_replace, col
import umap

# fixes for displaying long/wide dataframes
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', -1)
#np.set_printoptions(threshold=np.inf)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b22e8aaf-db58-4baf-be4d-0eb1d7ea4080"),
    inpatient_encoded=Input(rid="ri.foundry.main.dataset.cef3c32e-767c-4f6a-b669-3920dac46a10")
)
def data_by_site( inpatient_encoded):
    df = inpatient_encoded
    sites_with_values = df.groupby('data_partner_id').count()
    return sites_with_values.reset_index()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.64abd514-65b6-42f8-8075-0e63f23fcd0d"),
    inpatient_encoded_all_cols=Input(rid="ri.foundry.main.dataset.5d31d8ed-ed3e-4304-96f7-9cc2554ed092")
)
def data_by_site_all_cols( inpatient_encoded_all_cols):
    df = inpatient_encoded_all_cols
    sites_with_values = df.groupby('data_partner_id').count()
    return sites_with_values.reset_index()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.cef3c32e-767c-4f6a-b669-3920dac46a10"),
    inpatient_ml_dataset=Input(rid="ri.foundry.main.dataset.07927bca-b175-4775-9c55-a371af481cc1")
)
def inpatient_encoded(inpatient_ml_dataset):
    # get rid of ids, columns that are duplicates of other information,
    # or columns that are from the end of stay
    sdf = inpatient_ml_dataset
    sdf = sdf.drop('covid_status_name')
    sdf = sdf.drop('person_id')
    sdf = sdf.drop('visit_concept_id')
    sdf = sdf.drop('visit_concept_name')
    sdf = sdf.drop('visit_occurrence_id')
    sdf = sdf.drop('in_death_table')
    sdf = sdf.drop('severity_type')
    sdf = sdf.drop('length_of_stay')
    sdf = sdf.drop('ecmo')
    sdf = sdf.drop('aki_in_hospital')
    sdf = sdf.drop('invasive_ventilation')

    # these columns are 90% or greater NULL
    sdf = sdf.drop('miscellaneous_program', 'department_of_corrections', 'department_of_defense', 'other_government_federal_state_local_excluding_department_of_corrections', 'd-dimer_mg_l_feu', 'no_payment_from_an_organization_agency_program_private_payer_listed', 'procalcitonin_ng_ml', 'erythrocyte_sed_rate_mm_hr', 'bnp_pg_ml', 'nt_pro_bnp_pg_ml', 'medicaid', 'private_health_insurance', 'ferritin_ng_ml', 'medicare', 'payer_no_matching_concept')

    df = sdf.toPandas()

    # fixing columns so they work with sklearn
    df['visit_start'] = pd.to_datetime(df.visit_start_date).astype('int64')
    df['visit_end'] = pd.to_datetime(df.visit_end_date).astype('int64')
    df = df.drop(columns=['visit_start_date', 'visit_end_date'])
    
    df = pd.concat([df.drop('gender_concept_name', axis=1), pd.get_dummies(df.gender_concept_name, prefix='gender', drop_first=True)], axis=1)
    df = pd.concat([df.drop('race', axis=1), pd.get_dummies(df.race, prefix='race', drop_first=True)], axis=1)
    df = pd.concat([df.drop('ethnicity', axis=1), pd.get_dummies(df.ethnicity, prefix='ethnicity', drop_first=True)], axis=1)
    df = pd.concat([df.drop('smoking_status', axis=1), pd.get_dummies(df.smoking_status, prefix='smoking', drop_first=True)], axis=1)
    df = pd.concat([df.drop('blood_type', axis=1), pd.get_dummies(df.blood_type, prefix='blood_type', drop_first=True)], axis=1)
    #df = pd.concat([df.drop('severity_type', axis=1), pd.get_dummies(df.severity_type, prefix='severity', drop_first=True)], axis=1)

    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_')
    df.columns = df.columns.str.lower()
    
    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5d31d8ed-ed3e-4304-96f7-9cc2554ed092"),
    inpatient_ml_dataset=Input(rid="ri.foundry.main.dataset.07927bca-b175-4775-9c55-a371af481cc1")
)
def inpatient_encoded_all_cols(inpatient_ml_dataset):
    sdf = inpatient_ml_dataset
    df = sdf.toPandas()

    # fixing columns so they work with sklearn
    df['visit_start'] = pd.to_datetime(df.visit_start_date).astype('int64')
    df['visit_end'] = pd.to_datetime(df.visit_end_date).astype('int64')
    df = df.drop(columns=['visit_start_date', 'visit_end_date'])
    
    #dummy code these
    df = pd.concat([df.drop('covid_status_name', axis=1), pd.get_dummies(df.covid_status_name, prefix='cov_status', drop_first=True)], axis=1)
    df = pd.concat([df.drop('gender_concept_name', axis=1), pd.get_dummies(df.gender_concept_name, prefix='gender', drop_first=True)], axis=1)
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
    Output(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    inpatient_encoded=Input(rid="ri.foundry.main.dataset.cef3c32e-767c-4f6a-b669-3920dac46a10")
)
def inpatient_scaled_w_imputation(inpatient_encoded):
    df = inpatient_encoded

    prediction = df.bad_outcome
    filled_df = df.drop(columns='bad_outcome')

    # this is bad, but just fill all nulls with median
    # try mo'bettah imputation later
    filled_df = filled_df.fillna(filled_df.median())

    scaler = StandardScaler()
    scaler.fit(filled_df)

    ret_df = pd.DataFrame(scaler.transform(filled_df), columns=filled_df.columns)
    ret_df['bad_outcome'] = prediction
    return ret_df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.05f810fb-9481-4a6c-9ef1-ea81d5d93476"),
    inpatient_encoded=Input(rid="ri.foundry.main.dataset.cef3c32e-767c-4f6a-b669-3920dac46a10")
)
def missing_data_info(inpatient_encoded):
    df = inpatient_encoded
    missing_df = df.isnull().sum().to_frame()
    missing_df = missing_df.rename(columns = {0:'null_count'})
    missing_df['pct_missing'] = missing_df['null_count'] / df.shape[0]
    missing_df = missing_df.reset_index()
    missing_df = missing_df.rename(columns = {'index':'variable'})
    missing_df = missing_df.sort_values('pct_missing', ascending=False)
    return missing_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.71dc050c-5c6a-442b-8b87-31b45277459a"),
    inpatient_encoded_all_cols=Input(rid="ri.foundry.main.dataset.5d31d8ed-ed3e-4304-96f7-9cc2554ed092")
)
def missing_data_info_all_cols(inpatient_encoded_all_cols):
    df = inpatient_encoded_all_cols
    missing_df = df.isnull().sum().to_frame()
    missing_df = missing_df.rename(columns = {0:'null_count'})
    missing_df['pct_missing'] = missing_df['null_count'] / df.shape[0]
    missing_df = missing_df.reset_index()
    missing_df = missing_df.rename(columns = {'index':'variable'})
    missing_df = missing_df.sort_values('pct_missing', ascending=False)
    return missing_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e6967b18-d64f-4539-9a9f-7ae3a5eef700"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2")
)
def pca3_ranked_features( inpatient_scaled_w_imputation):
    df = inpatient_scaled_w_imputation
    prediction = df.bad_outcome
    # take out prediction column
    df = df.drop(columns='bad_outcome')
    scaled_arr = df.values

    # now the top 3 viewed with outcome
    pca_3 = PCA(n_components=3, random_state=42)
    pca_3.fit(scaled_arr)
    pca_3_arr = pca_3.transform(scaled_arr)

    pca_df = pd.DataFrame(pca_3.components_, columns=df.columns,index = ['PC-1','PC-2', 'PC-3'])
    pt = pca_df.transpose().abs()
    pc1_df = pt.sort_values('PC-1', ascending=False).drop(['PC-2', 'PC-3'], axis=1)
    pc1_df = pc1_df.reset_index()
    pc1_df = pc1_df.rename(columns = {'index':'pc-1-col', 'PC-1': 'pc-1-val'})

    pc2_df = pt.sort_values('PC-2', ascending=False).drop(['PC-1', 'PC-3'], axis=1)
    pc2_df = pc2_df.reset_index()
    pc2_df = pc2_df.rename(columns = {'index':'pc-2-col', 'PC-2': 'pc-2-val'})

    pc3_df = pt.sort_values('PC-3', ascending=False).drop(['PC-1', 'PC-2'], axis=1)
    pc3_df = pc3_df.reset_index()
    pc3_df = pc3_df.rename(columns = {'index':'pc-3-col', 'PC-3': 'pc-3-val'})

    sdf = spark.createDataFrame(pd.concat([pc1_df, pc2_df, pc3_df], axis=1))
    sdf = sdf.sort(col('pc-1-val').desc())

    return sdf

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e60a9fc6-c946-4707-a1a0-bea11453ad48"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2")
)
def pca_2_comp_analysis( inpatient_scaled_w_imputation):
    # decent PCA guide available here: https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
    df = inpatient_scaled_w_imputation
    prediction = df.bad_outcome
    # take out prediction column
    df = df.drop(columns='bad_outcome')
    scaled_arr = df.values

    #start with all variables for PCA
    my_pca = PCA(n_components=scaled_arr.shape[1], random_state=42)
    my_pca.fit(scaled_arr)
    pca_arr = my_pca.transform(scaled_arr)

    # now the top 3 viewed with outcome
    pca_2 = PCA(n_components=2, random_state=42)
    pca_2.fit(scaled_arr)
    pca_2_arr = pca_2.transform(scaled_arr)

    fig = plt.figure(figsize = (12, 8))

    splt = sns.scatterplot(x = pca_2_arr[:, 0],
                            y = pca_2_arr[:, 1],
                            s = 100,
                            hue = prediction,
                            alpha = 0.6)

    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    mytitle = 'PCA 2D scatter plot - ' + str(round(np.cumsum(my_pca.explained_variance_ratio_ * 100)[1])) + '% of variance captured'
    plt.title(mytitle)
    plt.show()

    # see https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
    return spark.createDataFrame(pd.DataFrame(pca_2.components_, columns=df.columns,index = ['PC-1','PC-2']).reset_index())

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a230c6e9-ece6-46e0-89aa-c9414533899f"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2")
)
def pca_3_comp_analysis( inpatient_scaled_w_imputation):
    # decent PCA guide available here: https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
    df = inpatient_scaled_w_imputation
    prediction = df.bad_outcome
    # take out prediction column
    df = df.drop(columns='bad_outcome')
    scaled_arr = df.values

    #start with all variables for PCA
    my_pca = PCA(n_components=df.shape[1], random_state=42)
    my_pca.fit(scaled_arr)
    pca_arr = my_pca.transform(scaled_arr)

    # now the top 3 viewed with outcome
    pca_3 = PCA(n_components=3, random_state=42)
    pca_3.fit(scaled_arr)
    pca_3_arr = pca_3.transform(scaled_arr)

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
    mytitle = 'PCA 3D scatter plot - ' + str(round(np.cumsum(my_pca.explained_variance_ratio_ * 100)[2])) + '% of variance captured'
    plt.title(mytitle)
    plt.show()

    # see https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
    return spark.createDataFrame(pd.DataFrame(pca_3.components_, columns=df.columns,index = ['PC-1','PC-2', 'PC-3']).reset_index())

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.78eb8376-28de-4705-b6b1-d5d2cf520b45"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2")
)
def pca_3_dataset(inpatient_scaled_w_imputation):
    # decent PCA guide available here: https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
    df = inpatient_scaled_w_imputation
    prediction = df.bad_outcome
    # take out prediction column
    df = df.drop(columns='bad_outcome')
    scaled_arr = df.values

    # now the top 3 viewed with outcome
    pca_3 = PCA(n_components=3, random_state=42)
    pca_3.fit(scaled_arr)
    pca_3_arr = pca_3.transform(scaled_arr)

    return pd.DataFrame(pca_3_arr)

@transform_pandas(
    Output(rid="ri.vector.main.execute.17ab7d1e-d4f3-4a5f-98a7-f63f5997c021"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2")
)
def pca_explained_variance( inpatient_scaled_w_imputation):
    df = inpatient_scaled_w_imputation
    prediction = df.bad_outcome
    # take out prediction column
    df = df.drop(columns='bad_outcome')
    scaled_arr = df.values

    #start with all variables for PCA
    my_pca = PCA(n_components=scaled_arr.shape[1], random_state=42)
    my_pca.fit(scaled_arr)
    pca_arr = my_pca.transform(scaled_arr)

    plt.plot(np.cumsum(my_pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.727dca65-eb02-41a3-b741-343d7b848573"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2")
)
def umap_analysis( inpatient_scaled_w_imputation):
    df = inpatient_scaled_w_imputation
    # try with just a few variables from the PCA analysis
    #df = df[['q_score', 'systolic_blood_pressure', 'visit_end', 'renal', 'respiratory_rate', 'visit_start', 'dmcx', 'negative_covid_test', 'negative_covid_test', 'dm', 'diastolic_blood_pressure', 'suspected_covid', 'chf', 'lactate_mg_dl', 'testcount']]
    #df = df.sample(frac=0.2)
    prediction = df.bad_outcome
    # take out prediction column
    df = df.drop(columns='bad_outcome')
    scaled_arr = df.values

    reducer = umap.UMAP(low_memory=True, random_state=42)
    reducer.fit(scaled_arr)
    embedding = reducer.transform(scaled_arr)

    #embedding = reducer.fit_transform(scaled_arr)

    print(embedding.shape)

    plt.scatter(embedding[:, 0],embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection', fontsize=20)
    plt.show()
    return


