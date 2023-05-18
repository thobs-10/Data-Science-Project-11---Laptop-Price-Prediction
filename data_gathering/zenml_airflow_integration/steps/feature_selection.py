import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split


from feature_engine.selection import DropCorrelatedFeatures
from sklearn.decomposition import PCA

from zenml.steps import Output, step
# read data from dataset preprocessed folder
def read_data():
    df = pd.read_parquet('C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Data Science Project 11 - Laptop Price Prediction\\dataset\\feature_engineered_data\\feature_engineered.parquet.gzip')
    return df

@step
def seperate_dataset()->Output(
    X=pd.DataFrame,
    copy_X = pd.DataFrame,
    y = pd.Series
):
    df = read_data()
    # from last task of feature engineering
    # df = ti.xcom(task_ids = 'drop_features')
    X = df.drop(columns=['Price'])
    y = np.log(df['Price'])
    # make an original copy of X features( the ones without feature engineering)
    copy_X = X.copy()
    return X,copy_X,y
@step
def label_encode(X:pd.DataFrame)->Output(
   X=np.ndarray
    
):
   
    ct = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')
    X = ct.fit_transform(X)
    return X
# @step
# def feature_importance(X:np.ndarray,y:pd.Series)->Output(
#     selection=ExtraTreesRegressor
# ):
#     # from task label encode
#     # X, y = ti.xcom(task_ids = 'label_encode')
#     # Important feature using ExtraTreesRegressor
#     selection = ExtraTreesRegressor()
#     selection.fit(X, y)
#     return selection
@step
def drop_top_correlated_features(df_X:pd.DataFrame)->Output(
    to_drop=list,
    df_X=pd.DataFrame
):
    # from task label encode
    # X, y = ti.xcom(task_ids = 'label_encode')
    # removing correlated variables from dataframe using DropCorrelatedFeatures
    #original_X[['name','company','year','fuel_type','Mileage']].corr()

    # removing correlated features
    #df_X = pd.DataFrame(X)

    cor_matrix = df_X.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.80)]
    return to_drop, df_X
@step
def drop_correlated(to_drop:list, df_X:pd.DataFrame)->Output(
    X = pd.DataFrame
):
    # from task drop top correlated features
    # to_drop, df_X = ti.xcom(task_ids = 'drop_top_correlated_features')
    for col in df_X.columns:
        for i in to_drop:
            if col == i:
                df_X.drop(columns=[col],inplace = True)
                df_X.drop(col,axis=1,inplace = True)
    df_final = df_X.copy()
    X = pd.DataFrame(df_final)
    return X

@step
def split_for_PCA(X:np.ndarray,y:pd.Series)->Output(
    X_train=np.ndarray,
    X_test = np.ndarray,
    y_train = pd.Series,
    y_test = pd.Series
):
    # from task drop correlated 
    # X = ti.xcom(task_ids ='drop_correlated')
    # and from task label encode
    # wrong_X, y = ti.xcoms(task_ids = 'label_encode')
    # original y, Latest X
    #X_new = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    return X_train,X_test, y_train, y_test
# @step
# def principal_component_analysis( X_train:np.ndarray,X_test:np.ndarray,X:pd.DataFrame,copy_X:pd.DataFrame)->Output(
#     df=pd.DataFrame,
#     selected_x = pd.DataFrame,
#     most_important_names = list
# ):
#     # x train and x test from task split for PCA
#     # X_train,X_test = ti.xcom(task_ids ='split_for_PCA')
#     # get X from the drop correlated task
#     # X = ti.xcom(task_ids ='drop_correlated')
#     # get the copy of X from the seperate dstaset  task
#     # wrong_X,copy_X,y = ti.xcom(task_ids = 'seperate_data')
#     #X_train_new = np.ndarray(X_train)
#     pca = PCA(n_components = copy_X.shape[1])
#     X_train = pca.fit_transform(X_train)
#     X_test = pca.transform(X_test)
#     # number of components
#     n_pcs= pca.components_.shape[0]
#     #n_pcs
#     # get the index of the most important feature on EACH component i.e. largest absolute value
#     # using LIST COMPREHENSION HERE
#     most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
#     #most_important
#     #initial_feature_names = ['name','company','year','fuel_type','Mileage']
#     initial_feature_names = list(copy_X.columns)
#     # get the names
#     most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
#     # using LIST COMPREHENSION HERE AGAIN
#     dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
#     # build the dataframe
#     df = pd.DataFrame(sorted(dic.items()))
#     #df
#     # get all the selected features from the dataframe
#     X_new = df.iloc[:, -1].values
#     #list(X_new)
#     # get selected features of the PCA
#     selected_x = X.loc[:, list(X_new)]
#     return df, selected_x, most_important_names
# @step
# def get_most_important_features(most_important_names:list,X:pd.DataFrame,y:pd.Series)->Output(
#     X_train=np.ndarray,
#     X_test = np.ndarray,
#     y_train = np.ndarray,
#     y_test = np.ndarray
# ):
#     # most importance names from the PCA task
#     # df, selected_x, most_important_names = ti.xcom(task_ids='pca')
#     # get the original X features and y feature, X from drop_top_correlated_features
#     # X, copy_X, y = ti.xcom(task_ids ='drop_top_correlated_features')
#     # the names of chosen X are based on the selected features of x that have the most impact
#     chosen_X = X[most_important_names]
#     #chosen_X = X[['Mileage', 'name', 'company', 'year']]
#     # the final spliting of data to train and test based on the most important features of X
#     X_train, X_test, y_train, y_test = train_test_split(chosen_X, y, test_size = 0.2, random_state = 1)
#     return X_train, X_test, y_train, y_test
@step
def feature_scaling( X_train:np.ndarray, X_test:np.ndarray, y_train:pd.Series, y_test:pd.Series)->Output(
    X_train_sc = np.ndarray,
    X_test_sc = np.ndarray,
    y_train = pd.Series,
    y_test = pd.Series
):
    # get values from the get most important features task
    # X_train, X_test, y_train, y_test = ti.xcom(task_ids='get_most_important_features')
    sc = StandardScaler()
    X_train_new = np.array(X_train)
    X_test_new = np.array(X_test)
    X_train_sc = sc.fit_transform(X_train_new)
    X_test_sc = sc.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test
