import pandas as pd
import numpy as np
import joblib, os, ray, glob
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.graph_objects as go
from itertools import combinations
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                    StratifiedKFold, StratifiedShuffleSplit,
                                    cross_val_score)
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, classification_report
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def line(x):
        return ((3.6 - 3.2) / (5000 - 6000)) * (x - 6000) + 3.2     # taken from literature

def calculate_boundary_lines(dfr, x, y, x1, y1, x2, y2, x3, y3, x4, y4):
    def line_above(x):
        if (x >= x2) and (x <= x1):  # for the first line
            return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1
        elif (x >= x3) and (x <= x2):  # for the second line
            return ((y3 - y2) / (x3 - x2)) * (x - x2) + y2

    def line_below(x):
        return ((y4 - y1) / (x4 - x1)) * (x - x4) + y4

    dfr['above_line'] = dfr[y] < dfr[x].apply(line_above)
    dfr['below_line'] = dfr[y] > dfr[x].apply(line_below)

    # Adds the column that identifies Giants and dwarfs as per requested by Dr carlos
    #i suppose as the petition stated that i needed to use the boundary_line i just put it in here
    df_original['Line_Label'] = np.where((df_original[x] >= 3000) & (df_original[x] <= 6000) & dfr['above_line'] & dfr['below_line'], 'Giants', 'Dwarfs')

    selected_points = dfr[(dfr[x] >= 3000) & (dfr[x] <= 6000) & dfr['above_line'] & dfr['below_line']]

    return dfr, selected_points, ((x1, y1), (x2, y2)), ((x2, y2), (x3, line_above(x3))),((x3, line_above(x3)), (x3, y4)), ((x3, y4), (x4, y4)), ((x4, y4), (x1, y1))


def optimize_hyperparameters(X, y):
    # Initialize a BalancedRandomForestClassifier
    clf = RandomForestRegressor()
    # Define hyperparameters grid
    param_grid = {
        'n_estimators': [200,500,800],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 30, None]    }
    # Grid search with cross-validation
    grid_search = GridSearchCV(estimator=RandomForestRegressor(),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',  # For regression
                           cv=10,  # Number of cross-validation folds
                           verbose=2,
                           n_jobs=-1  # Use all available CPUs
                          ) 
    grid_search.fit(X, y)
    # Return best parameters and best score
    return grid_search



def preprocess_features(data,columns,bad_values):
    """
    Realiza el preprocesamiento de las características especificadas.

    Args:
    - data (pd.DataFrame): DataFrame original con los datos.

    Returns:
    - data (pd.DataFrame): DataFrame procesado con características adicionales y filas inválidas eliminadas.
    """
    
    # Busca valores 999 o -999 en las columnas especificadas
    invalid_values = data[columns].isin([bad_values, -bad_values]).sum()
    # Filtra y muestra solo las columnas que tienen estos valores inválidos
    columns_with_invalids = invalid_values[invalid_values > 0]
    if columns_with_invalids.empty:
        print(f" Todas las columnas tienen datos válidos, sin valores {bad_values} o -{bad_values}.")
    else:
        print(f"Las siguientes columnas tienen valores {bad_values} o -{bad_values}:\n{columns_with_invalids}")
    # Filtra el DataFrame para excluir filas con valores 999 o -999 en las columnas especificadas
    data = data[~data[columns].isin([bad_values, -bad_values]).any(axis=1)].reset_index(drop=True)
    return data



class DataFrameScaler:
    def __init__(self):
        self.scalers = {}
    def scale(self, df):
        """
        Escala las columnas del DataFrame entre Min y Max 
        """
        scaled_df = df.copy()
        for column in df.columns:
            scaler = MinMaxScaler()
            scaled_df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1)).flatten()
            self.scalers[column] = scaler
        return scaled_df
    def inverse_scale(self, df):
        """
        Devuelve el DataFrame escalado a su estado original.
        """
        original_df = df.copy()
        for column in df.columns:
            scaler = self.scalers.get(column)
            if scaler:
                original_df[column] = scaler.inverse_transform(df[column].values.reshape(-1, 1)).flatten()
        return original_df


def create_color_columns(mag_features, df):
    """
    Crea columnas de colores en el DataFrame df basadas en las combinaciones de mag_features.
    
    Parámetros:
    - mag_features: Lista de características para hacer combinaciones y crear colores.
    - df: DataFrame donde se añadirán las nuevas columnas de colores.
     
    Retorna:
    - df: DataFrame modificado con las nuevas columnas de colores.
    - colors: Lista de los nombres de las nuevas columnas de colores.
    """
    colors = []
    for v in combinations(mag_features, 2):
        df[f"{v[0]}-{v[1]}"] = df[v[0]] - df[v[1]]
        colors.append(f"{v[0]}-{v[1]}")
    return df, colors

def exclusion_criteria(data):
    #the parameters suggest that only the values that attend to the criteria are accounted for and used to return to data
    #it should work as it only using np functions if it's not the parameters that've been asked it's adjustable on the 3*deviation
    for column in data.columns:
        deviation = np.std(data[column])
        absolute_values_within_3std = np.abs(data[column]) <= 3 * deviation
        data[column] = np.where(absolute_values_within_3std, data[column], np.nan)
    return data
    


    
def process_columns(df_original, columns_name, colors, survey, idx,giagiants):
    for column in columns_name[idx]:
        column_save = column.replace("[","").replace("]","").replace("/","")
        test = survey + '_' + str(column_save)
        # Crear directorios para los resultados
        directory_path = f'/home/carloslopes/Documents/splus2023/resultsRF/{giagiants}/{survey}/{column_save}'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        AnasCores = df_original.copy(deep=True)
        x_columns = colors
        y_column = [column]
        print(y_column)
        # Preprocesamiento
        AnasCores = preprocess_features(AnasCores, AnasCores[x_columns + y_column].columns, 999)
        AnasCores = preprocess_features(AnasCores, AnasCores[x_columns + y_column].columns, 9999)
        AnasCores = preprocess_features(AnasCores, AnasCores[x_columns + y_column].columns, np.nan)
        # División de datos
        X_train, X_test, Y_train, Y_test = train_test_split(AnasCores[x_columns], AnasCores[column], 
                                                            test_size=0.2, random_state=11085)
        # Optimización de hiperparámetros
        grid_search = optimize_hyperparameters(X_train, Y_train)
        best_params, best_score = grid_search.best_params_, grid_search.best_score_
        params_file = os.path.join(directory_path, f'best_param_{test}.csv')
        pd.DataFrame([best_params]).to_csv(params_file, index=False)
        # Entrenamiento del modelo
        rf_regressor = RandomForestRegressor(**best_params)
        rf_regressor.fit(X_train, Y_train)        
        #Guardar dataframe
        AnasCores.loc[X_train.index,"split"] = "train"
        AnasCores.loc[X_test.index,"split"] = "test"
        df_file = os.path.join(directory_path, f'df_{test}.csv')
        AnasCores.to_csv(df_file, index=False)
        # Guardar el modelo entrenado
        model_file = os.path.join(directory_path, f'{test}.sav')
        joblib.dump(rf_regressor, model_file)

def plot_data(df, selected_points, additional_lines, xcol, ycol, title):
    plt.figure(figsize=(10, 8))    
    for line_start, line_end in additional_lines:
        plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'g--')
    sns.scatterplot(x=df[xcol], y=df[ycol], s=2, color='red')
    sns.scatterplot(x=selected_points[xcol], y=selected_points[ycol], s=2, color='blue')
    plt.xlabel('Teff')
    plt.ylabel('logG')
    plt.title(title)
    plt.ylim(5, 0)
    plt.xlim(7500, 3000)
    plt.tight_layout()
    plt.savefig(title+'.png', dpi=300)
    plt.show()



# ----------------- change only this section ----------------------
xcolc = 'Teff'               # define x-column
ycolc = 'logg'               # define y-column
xc1, yc1 = 5800, 3.2            #starting pt of 1st line
xc2, yc2 = 4500, line(4500)     #ending pt of 1st line, starting pt of 2nd line
xc3, yc3 = 3020, 1              #ending pt of 2nd line, starting pt of 3rd line
xc4, yc4 = 4800, 0.05           #ending pt of 3rd line, starting pt of 4th line
# ------------------------------------------------------------------


#DEFINE PARAMETERS AND RUNNING PREDICTION
survey = ["Apogee","Galah","LamostMedium"]
mpath = '/home/carloslopes/Documents/splus2023/'
file_names = ["Apogee",
    "Galah","LamostMedium"]

apoge_columns = ['TEFF', 'LOGG', 'FE_H', 'ALPHA_M', 'C_FE', 'CA_FE', 'N_FE',
       'NI_FE', 'MG_FE', 'SI_FE']
galah_columns = ['Teff', 'logg', '[Fe/H]', '[alpha/Fe]', '[Li/Fe]',
       '[C/Fe]', '[O/Fe]', '[Na/Fe]', '[Mg/Fe]', '[Al/Fe]', '[Si/Fe]',
       '[K/Fe]', '[Ca/Fe]', '[Sc/Fe]', '[Ti/Fe]', '[Ti2/Fe]',
        '[Cr/Fe]', '[Mn/Fe]', '[Co/Fe]', '[Ni/Fe]',
       '[Cu/Fe]', '[Zn/Fe]', '[Y/Fe]']
lamostMedium_columns = ['teff_cnn', 'logg_cnn', 'feh_cnn', 'alpha_m_cnn', 'c_fe', 'ca_fe',
       'n_fe', 'mg_fe', 'si_fe', 'ni_fe']
lamostLow_columns = ['teff', 'logg', 'feh',"alpha_m"]
columns_name = [apoge_columns,galah_columns,lamostMedium_columns,lamostLow_columns]
s_plus_filter = ['u', 'J378', 'J395', 'J410',
         'J430', 'g', 'J515', 'r', 'J660', 'i', 'J861', 'z']


for idx,survey in enumerate (file_names for idx, survey in enumerate(file_names):):
    df_original = pd.read_csv(glob.glob(f'{mpath}trainset/{survey}*')[0])
    df_original = preprocess_features(df_original,s_plus_filter,999)
    df_original = preprocess_features(df_original,s_plus_filter,9999)
    df_original = preprocess_features(df_original,s_plus_filter,np.nan)
    df_original = df_original.loc[((df_original[s_plus_filter] < 20).all(axis=1))&
               ((df_original[s_plus_filter] > 10 ).all(axis=1))]
    df_original,colors = create_color_columns(s_plus_filter, df_original)

    scaler = DataFrameScaler()
    df_original[colors] = scaler.scale(df_original[colors])
    df_original= exclusion_criteria(df_original)

    process_columns(df_original.loc[df_original["giagiants"]==1], columns_name, colors, survey, idx,"giagiants")
    process_columns(df_original.loc[df_original["giagiants"]!=1], columns_name, colors, survey, idx,"dwarfs")


df = df_galah.copy()        

dfr, selected_points, *additional_lines = calculate_boundary_lines(df_original, xcolc, ycolc, xc1, yc1, xc2, yc2, xc3, yc3, xc4, yc4)
df_original["split"] = [10, 20, 30, 40]

plot_data(dfr, selected_points, additional_lines, xcolc, ycolc, 'galah')


