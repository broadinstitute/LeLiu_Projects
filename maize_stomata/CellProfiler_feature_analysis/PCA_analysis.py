
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def PCA_analysis(df, scale):
   
    df = df.dropna()
    if scale == True:
        numeric_columns = df.select_dtypes(include='number')
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(numeric_columns)
        df[numeric_columns.columns] = normalized_data
        pca = PCA()
        pca_result = pca.fit_transform(normalized_data)
        pca_columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
        pca_df = pd.DataFrame(data=pca_result, columns=pca_columns)
        final_df = pd.concat([df.drop(columns=numeric_columns.columns), pca_df], axis=1)
    else: 
        numeric_columns = df.select_dtypes(include='number')
        pca = PCA()
        pca_result = pca.fit_transform(numeric_columns)
        pca_columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
        pca_df = pd.DataFrame(data=pca_result, columns=pca_columns)
        final_df = pd.concat([df.drop(columns=numeric_columns.columns), pca_df], axis=1)
    return pca, final_df









