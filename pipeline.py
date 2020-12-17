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
    Output(rid="ri.foundry.main.dataset.e6967b18-d64f-4539-9a9f-7ae3a5eef700"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.bc823c17-fcdc-4801-a389-c6f476ed6971")
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
    pt = pca_df.transpose()
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
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.bc823c17-fcdc-4801-a389-c6f476ed6971")
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
    my_pca = PCA(n_components=df.shape[1], random_state=42)
    my_pca.fit(scaled_arr)
    pca_arr = my_pca.transform(scaled_arr)

    # now the top 2 viewed with outcome
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
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.bc823c17-fcdc-4801-a389-c6f476ed6971")
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
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.bc823c17-fcdc-4801-a389-c6f476ed6971")
)
def pca_all_dataset(inpatient_scaled_w_imputation):
    # decent PCA guide available here: https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
    df = inpatient_scaled_w_imputation
    
    # take out visit_occurrence_id column
    df = df.drop(columns='visit_occurrence_id')
    scaled_arr = df.values

    # now the top 3 viewed with outcome
    pca_all = PCA(random_state=42)
    pca_all.fit(scaled_arr)
    pca_all_arr = pca_all.transform(scaled_arr)

    return pd.DataFrame(pca_all_arr)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ad26ec53-2c46-4d3c-9a78-c86c77accad7"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.bc823c17-fcdc-4801-a389-c6f476ed6971")
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
    Output(rid="ri.foundry.main.dataset.d2ff4c8e-fbe2-445d-9df2-356a3a70e4f6"),
    pca_all_dataset=Input(rid="ri.foundry.main.dataset.78eb8376-28de-4705-b6b1-d5d2cf520b45")
)
def pca_umap2d_embedding( pca_all_dataset):
    scaled_arr = pca_all_dataset.iloc[:, :20]

    reducer = umap.UMAP(random_state=42, n_neighbors=400, local_connectivity=20)
    reducer.fit(scaled_arr)
    embedding = reducer.transform(scaled_arr)
    return pd.DataFrame(embedding)


@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d0d29902-d951-4b39-aacd-117a2c981f1e"),
    pca_all_dataset=Input(rid="ri.foundry.main.dataset.78eb8376-28de-4705-b6b1-d5d2cf520b45")
)
def pca_umap2d_embedding_stdscaled(pca_all_dataset):
    arr = pca_all_dataset.iloc[:, :20]

    scaler = StandardScaler()
    scaler.fit(arr)
    scaled_arr = scaler.transform(arr)

    reducer = umap.UMAP(random_state=42, n_neighbors=200, local_connectivity=5)
    reducer.fit(scaled_arr)
    embedding = reducer.transform(scaled_arr)
    return pd.DataFrame(embedding)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f7acb861-abfd-47a4-951a-4d0735ad9cf8"),
    pca_umap2d_embedding_stdscaled=Input(rid="ri.foundry.main.dataset.d0d29902-d951-4b39-aacd-117a2c981f1e")
)
def pca_umap2d_scaled_viz_bad_outcome( outcomes, pca_umap2d_embedding_stdscaled):
    embedding = pca_umap2d_embedding_stdscaled.values
    dfo = outcomes

    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.bad_outcome,
                            alpha = 0.6)
    plt.title('PCA UMAP 2D scatter plot')
    plt.show()
    
    return



@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0636cf0d-202c-4919-bc65-06577254b95e"),
    pca_umap2d_embedding_stdscaled=Input(rid="ri.foundry.main.dataset.d0d29902-d951-4b39-aacd-117a2c981f1e")
)
def pca_umap2d_scaled_viz_severity( outcomes, pca_umap2d_embedding_stdscaled):
    embedding = pca_umap2d_embedding_stdscaled.values
    dfo = outcomes
    dfo['severity_type'] = dfo.severity_type.astype('category')

    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.severity_type,
                            alpha = 0.6)
    plt.title('PCA UMAP 2D scatter plot')
    plt.show()
    
    return



@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6b6da43c-b691-45d8-a81a-b95def6bba59"),
    pca_umap2d_embedding_stdscaled=Input(rid="ri.foundry.main.dataset.d0d29902-d951-4b39-aacd-117a2c981f1e")
)
def pca_umap2d_scaled_viz_site( outcomes, pca_umap2d_embedding_stdscaled):
    embedding = pca_umap2d_embedding_stdscaled.values
    dfo = outcomes
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.data_partner_id,
                            alpha = 0.6,
                            legend = True)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")                            
    plt.title('PCA UMAP 2D scatter plot')
    plt.show()
    
    return



@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0b242849-8c9f-495c-955d-053da3a22e43"),
    pca_umap2d_embedding=Input(rid="ri.foundry.main.dataset.d2ff4c8e-fbe2-445d-9df2-356a3a70e4f6")
)
def pca_umap2d_viz_bad_outcome( outcomes, pca_umap2d_embedding):
    embedding = pca_umap2d_embedding.values
    dfo = outcomes
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.bad_outcome,
                            alpha = 0.6)
    plt.title('PCA UMAP 2D scatter plot')
    plt.show()
    
    return



@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4f9b9075-269e-4aa6-bc77-d56491bce711"),
    pca_umap2d_embedding=Input(rid="ri.foundry.main.dataset.d2ff4c8e-fbe2-445d-9df2-356a3a70e4f6")
)
def pca_umap2d_viz_severity( outcomes, pca_umap2d_embedding):
    embedding = pca_umap2d_embedding.values
    dfo = outcomes
    dfo['severity_type'] = dfo.severity_type.astype('category')

    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.severity_type,
                            alpha = 0.6)
    plt.title('PCA UMAP 2D scatter plot')
    plt.show()
    
    return



@transform_pandas(
    Output(rid="ri.foundry.main.dataset.50e500ee-112f-41d6-8340-886b40d61b39"),
    pca_umap2d_embedding=Input(rid="ri.foundry.main.dataset.d2ff4c8e-fbe2-445d-9df2-356a3a70e4f6")
)
def pca_umap2d_viz_site( outcomes, pca_umap2d_embedding):
    embedding = pca_umap2d_embedding.values
    dfo = outcomes
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    #fig = plt.figure(figsize = (12, 8))
    #ax = plt.axes()
    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.data_partner_id,
                            alpha = 0.6,
                            legend = True)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.title('PCA UMAP 2D scatter plot')
    plt.show()
    
    return



@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ba772263-cc7a-41ab-82a2-0203139bbbf4"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.bc823c17-fcdc-4801-a389-c6f476ed6971")
)
def umap2d_embedding(inpatient_scaled_w_imputation):
    df = inpatient_scaled_w_imputation
    df = df.drop(columns='visit_occurrence_id')
    scaled_arr = df.values

    reducer = umap.UMAP(random_state=42, n_neighbors=500, local_connectivity=5)
    reducer.fit(scaled_arr)
    embedding = reducer.transform(scaled_arr)
    return pd.DataFrame(embedding)


@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fd3540d6-d8ec-4374-a8d5-a573bf1292f5"),
    umap2d_embedding=Input(rid="ri.foundry.main.dataset.ba772263-cc7a-41ab-82a2-0203139bbbf4")
)
def umap2d_viz_bad_outcome( outcomes, umap2d_embedding):
    embedding = umap2d_embedding.values
    dfo = outcomes
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.bad_outcome,
                            alpha = 0.6)
    plt.title('UMAP 2D scatter plot')
    plt.show()
    
    return


@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f9367601-0a80-489e-9958-4fd41a796713"),
    umap2d_embedding=Input(rid="ri.foundry.main.dataset.ba772263-cc7a-41ab-82a2-0203139bbbf4")
)
def umap2d_viz_severity_type( outcomes, umap2d_embedding):
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
    Output(rid="ri.foundry.main.dataset.41b56b11-253f-4396-9ce9-71e381af695f"),
    umap2d_embedding=Input(rid="ri.foundry.main.dataset.ba772263-cc7a-41ab-82a2-0203139bbbf4")
)
def umap2d_viz_site( outcomes, umap2d_embedding):
    embedding = umap2d_embedding.values
    dfo = outcomes
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.data_partner_id,
                            alpha = 0.6)
    plt.title('UMAP 2D scatter plot')
    plt.show()
    
    return


@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c135a77f-4b71-4df9-abfe-be348abfc6a8"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.bc823c17-fcdc-4801-a389-c6f476ed6971")
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
                        opacity=0.6,
                        title="UMAP 3D by Bad Outcome")
    fig.show()
    
    return

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4e12ebec-cf89-4018-813f-0ceefda14c1a"),
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
                        opacity=0.1,
                        title="UMAP 3D by Severity Type")
    fig.show()
    
    return

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f1c093ec-5b1f-438a-894d-eb7acd69d8c9"),
    umap3d_embedding=Input(rid="ri.foundry.main.dataset.c135a77f-4b71-4df9-abfe-be348abfc6a8")
)
def umap3d_viz_site(umap3d_embedding, outcomes):
    embedding = umap3d_embedding.values
    dfo = outcomes
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    fig = px.scatter_3d(x=embedding[:, 0],
                        y=embedding[:, 1],
                        z=embedding[:, 2],
                        color=dfo.data_partner_id,
                        opacity=0.6,
                        title="UMAP 3D by Site")
    fig.show()
    
    return

