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
import plotly.express as px
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
    sdf = sdf.drop('in_death_table')
    sdf = sdf.drop('severity_type')
    sdf = sdf.drop('length_of_stay')
    sdf = sdf.drop('ecmo')
    sdf = sdf.drop('aki_in_hospital')
    sdf = sdf.drop('invasive_ventilation')
    sdf = sdf.drop('bad_outcome')

    # these columns are 90% or greater NULL
    sdf = sdf.drop('miscellaneous_program', 'department_of_corrections', 'department_of_defense', 'other_government_federal_state_local_excluding_department_of_corrections', 'no_payment_from_an_organization_agency_program_private_payer_listed', 'medicaid', 'private_health_insurance')
    
    df = sdf.toPandas()

    # fixing columns so they work with sklearn
    df['visit_start'] = pd.to_datetime(df.visit_start_date).astype('int64')
    df['visit_end'] = pd.to_datetime(df.visit_end_date).astype('int64')
    df = df.drop(columns=['visit_start_date', 'visit_end_date'])
    
    df = pd.concat([df, pd.get_dummies(df.data_partner_id, prefix='site', drop_first=True)], axis=1)
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
    Output(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    inpatient_encoded=Input(rid="ri.foundry.main.dataset.cef3c32e-767c-4f6a-b669-3920dac46a10")
)
def inpatient_encoded_w_imputation(inpatient_encoded):
    df = inpatient_encoded
    df = df.drop(columns='data_partner_id')

    df['bnp_pg_ml'] = df['bnp_pg_ml'].fillna(100)
    df['c-reactive_protein_crp_mg_l'] = df['c-reactive_protein_crp_mg_l'].fillna(10)
    df['erythrocyte_sed_rate_mm_hr'] = df['erythrocyte_sed_rate_mm_hr'].fillna(19)
    df['lactate_mg_dl'] = df['lactate_mg_dl'].fillna(13.5)
    df['nt_pro_bnp_pg_ml'] = df['nt_pro_bnp_pg_ml'].fillna(125)
    df['procalcitonin_ng_ml'] = df['procalcitonin_ng_ml'].fillna(0.02)
    df['troponin_all_types_ng_ml'] = df['troponin_all_types_ng_ml'].fillna(0.02)

    df.loc[(df.gender_male == True) & (df.ferritin_ng_ml.isna()), 'ferritin_ng_ml'] = 150
    df.loc[(df.gender_male == False) & (df.gender_no_matching_concept == False) & (df.ferritin_ng_ml.isna()), 'ferritin_ng_ml'] = 75
    
    # fill these with False
    df['medicare'] = df['medicare'].fillna(False)
    df['payer_no_matching_concept'] = df['payer_no_matching_concept'].fillna(False)

    # now fill the rest with the median
    df = df.fillna(df.median())

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd")
)
def inpatient_scaled_w_imputation( inpatient_encoded_w_imputation):
    df = inpatient_encoded_w_imputation
    
    # this column should not be centered/scaled
    visit_occurrence_id = df['visit_occurrence_id']
    df = df.drop(columns='visit_occurrence_id')

    scaler = StandardScaler()
    scaler.fit(df)

    ret_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    ret_df['visit_occurrence_id'] = visit_occurrence_id
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
    Output(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c"),
    inpatient_ml_dataset=Input(rid="ri.foundry.main.dataset.07927bca-b175-4775-9c55-a371af481cc1")
)
def outcomes(inpatient_ml_dataset):
    df = inpatient_ml_dataset
    df = df.select('visit_occurrence_id',
                   'person_id',
                   'data_partner_id',
                   'visit_concept_name',
                   'covid_status_name',
                   'in_death_table',
                   'severity_type',
                   'length_of_stay',
                   'ecmo',
                   'aki_in_hospital',
                   'invasive_ventilation',
                   'bad_outcome')
    return df.toPandas()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e6967b18-d64f-4539-9a9f-7ae3a5eef700"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2")
)
def pca3_ranked_features( inpatient_scaled_w_imputation):
    dfo = inpatient_scaled_w_imputation
    # take out visit_occurrence_id column
    df = dfo.drop(columns='visit_occurrence_id')
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
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def pca_2_comp_analysis( inpatient_scaled_w_imputation, outcomes):
    # decent PCA guide available here: https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
    df = inpatient_scaled_w_imputation
    dfo = outcomes
    prediction = dfo.bad_outcome
    # take out visit_occurrence_id column
    df = df.drop(columns='visit_occurrence_id')
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
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def pca_3_comp_analysis( inpatient_scaled_w_imputation, outcomes):
    # decent PCA guide available here: https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
    df = inpatient_scaled_w_imputation
    dfo = outcomes
    
    df = df.drop(columns='visit_occurrence_id')
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
        c = dfo['bad_outcome'],
        s=50,
        alpha=0.6)

    ax.legend(*splt.legend_elements(), title='bad_outcome')

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
    
    # take out visit_occurrence_id column
    df = df.drop(columns='visit_occurrence_id')
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
    
    # take out visit_occurrence_id column
    df = df.drop(columns='visit_occurrence_id')
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
    Output(rid="ri.foundry.main.dataset.ba772263-cc7a-41ab-82a2-0203139bbbf4"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2")
)
def umap2d_embedding(inpatient_scaled_w_imputation):
    df = inpatient_scaled_w_imputation
    df = df.drop(columns='visit_occurrence_id')
    scaled_arr = df.values

    reducer = umap.UMAP(random_state=42)
    reducer.fit(scaled_arr)
    embedding = reducer.transform(scaled_arr)
    return pd.DataFrame(embedding)


@transform_pandas(
    Output(rid="ri.vector.main.execute.727dca65-eb02-41a3-b741-343d7b848573"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c"),
    umap2d_embedding=Input(rid="ri.foundry.main.dataset.ba772263-cc7a-41ab-82a2-0203139bbbf4")
)
def umap2d_viz( outcomes, umap2d_embedding):
    embedding = umap2d_embedding.values
    dfo = outcomes
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.severity_type,
                            alpha = 0.6)
    plt.title('UMAP 2D scatter plot')
    plt.show()
    
    return


@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c135a77f-4b71-4df9-abfe-be348abfc6a8"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2")
)
def umap3d_embedding(inpatient_scaled_w_imputation):
    df = inpatient_scaled_w_imputation
    df = df.drop(columns='visit_occurrence_id')
    scaled_arr = df.values

    reducer = umap.UMAP(n_components=3, random_state=42)
    reducer.fit(scaled_arr)
    embedding = reducer.transform(scaled_arr)
    return pd.DataFrame(embedding)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.58ae8bdf-979b-4847-ac9f-fdd4071c07ef"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c"),
    umap3d_embedding=Input(rid="ri.foundry.main.dataset.c135a77f-4b71-4df9-abfe-be348abfc6a8")
)
def umap3d_viz_bad_outcome(umap3d_embedding, outcomes):
    embedding = umap3d_embedding.values
    dfo = outcomes
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    fig = px.scatter_3d(x=embedding[:, 0],
                        y=embedding[:, 1],
                        z=embedding[:, 2],
                        color=dfo.bad_outcome,
                        title="UMAP 3D by Bad Outcome")
    fig.show()
    
    return

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4e12ebec-cf89-4018-813f-0ceefda14c1a"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c"),
    umap3d_embedding=Input(rid="ri.foundry.main.dataset.c135a77f-4b71-4df9-abfe-be348abfc6a8")
)
def umap3d_viz_severity_type(umap3d_embedding, outcomes):
    embedding = umap3d_embedding.values
    dfo = outcomes
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    fig = px.scatter_3d(x=embedding[:, 0],
                        y=embedding[:, 1],
                        z=embedding[:, 2],
                        color=dfo.severity_type,
                        title="UMAP 3D by Severity Type")
    fig.show()
    
    return

@transform_pandas(
    Output(rid="ri.vector.main.execute.a96babd6-ac73-4f4c-94dc-48040203759a"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c"),
    umap3d_embedding=Input(rid="ri.foundry.main.dataset.c135a77f-4b71-4df9-abfe-be348abfc6a8")
)
def umap3d_viz_severity_type_1(umap3d_embedding, outcomes):
    embedding = umap3d_embedding.values
    dfo = outcomes
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    fig = px.scatter_3d(x=embedding[:, 0],
                        y=embedding[:, 1],
                        z=embedding[:, 2],
                        color=dfo.severity_type,
                        title="UMAP 3D by Severity Type")
    fig.show()
    
    return

