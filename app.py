# importing libraries

import streamlit as st
from boston import * 
import shap
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title='Boston House Price Prediction', layout='wide')

# give description and title of the project
st.write("""
# Boston House Price Prediction
         
A web application to predict and analyze Boston House Price dataset.
Powered by `streamlit`.
         
For more information about the dataset, check [this website](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
""")
st.divider()

# use sidebar to layout-ing the website
sidebar = ('Exploratory Data Analysis', 'Machine Learning Boston House Prediction')
with st.sidebar:
    st.markdown('##### Author : Rasyid Sulaeman')
    st.markdown('##### Check out the repo [here](https://github.com/rasyidsulaeman/Boston_House_Price_Analysis)')
    st.header('Select Data Processess')
    radio = st.radio('Modes', options=sidebar)

# load dataset
path = 'dataset/boston.csv'
boston = BostonEDA(path)
df = boston.data_loaded()
df = df.astype('float64')

X = df.drop(columns='MEDV')
Y = df['MEDV']

if radio == sidebar[0]:

    # display the dataset
    st.write('#### Boston Dataset')
    st.dataframe(df) 

    # display general info : total columns, total entries
    st.write('#### General Info')
    cols_1, cols_2 = st.columns(2)
    cols_1.metric('Total columns', len(df.columns))
    cols_2.metric('Total entries', len(df))

    # display dataset properties
    st.dataframe(df.describe())

    # visualize correlation matrix
    st.write("""
    #### Correlation Matrix
            
    Visualize the strength and diretion of linear relationship between two columns
    """)
    st.write(boston.heatmap(df))

    # visualize the dataset
    st.write("""
    #### Data Visualization
    
    **Select data visualization you want to display.**
    
    1. **Histogram** : to understand normality properties of dataset 
    2. **Regression-Scatter** : to display linearity relationship between the attributes and the target value
    3. **Box** : to show data outlier in each feature
    """)
    option = boston.plot_option
    choose = st.selectbox('Choose following options', options=option)
    with st.spinner('Wait for it . . .'):
        st.write(boston.plots(df, choose))
    st.success('Done!')

    st.write("""
    #### Outlier Detection
    
    Compute number of outlier for each attribute along with its percentage and display them using table; sorted from highest outlier to lowest.
    """)
    outlier = boston.outlier_calculation(df)
    st.dataframe(outlier)

elif radio == sidebar[1]:

    st.write('## Machine Learning - Random Forest Regression Model')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    st.write('#### Tune the hyperparameter')

    parameter_columns = st.columns(2)

    with parameter_columns[0]:
        n_estimators = st.slider('Number of estimators', 0, 500, 100)
        criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error'])
        max_features = st.slider('Max features', 1, 50, 1)
    with parameter_columns[1]:
        max_depth = st.slider('The maximum depth of the tree', 1, 200, 20) 
        min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
        min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    forest_regression = RandomForestRegressor(n_estimators=n_estimators,
                                              criterion=criterion,
                                              max_features=max_features,
                                              max_depth=max_depth,
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              random_state=42, n_jobs=-1)
    forest_regression.fit(X_train, y_train)

    y_pred = forest_regression.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2)

    st.write('Root mean squared error:')
    st.info(rmse)

    st.write('Model parameters')
    st.write(forest_regression.get_params(deep=True))

    
    reg_df = pd.DataFrame({'Observed' : y_test, 
                           'Predicted' : y_pred})

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter_plot = sns.regplot(x='Observed', y='Predicted', data=reg_df, ax=ax)
    text = f'R-squared: {r2:.2f}\nRMSE: {rmse:.2f}'
    ax.text(0.05, 0.9, text, transform=ax.transAxes, fontsize=12)
    scatter_plot.set_xlabel('Observed')
    scatter_plot.set_ylabel('Predicted')

    st.write('#### Predicted vs Observed Linear Relationship')
    st.pyplot(fig)
    
    # Explaining the model's predictions using SHAP values
    explainer = shap.Explainer(forest_regression, X_test)
    shap_values = explainer(X_test)

    st.write('#### Feature Importance') 
    st.markdown('Visualized feature contribution using `shap` package')
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    st.pyplot(fig)

    st.write('---')
    st.write('## Boston Price Prediction')

    st.write('#### Specified Input parameters')
    
    columns = st.columns(3)
    with columns[0]:
        CRIM = st.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
        ZN = st.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
        INDUS = st.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
        CHAS = st.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
        NOX = st.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
    
    with columns[1]:
        RM = st.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
        AGE = st.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
        DIS = st.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
        RAD = st.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
    
    with columns[2]:
        TAX = st.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
        PTRATIO = st.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
        BLACK = st.slider('BLACK', X.BLACK.min(), X.BLACK.max(), X.BLACK.mean())
        LSTAT = st.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
    
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'BLACK': BLACK,
            'LSTAT': LSTAT}
    
    features = pd.DataFrame(data, index=[0])
   
    # Print specified input parameters
    st.write('##### This is your defined parameters')
    st.write(features)
    st.write('---')

    # Apply Model to Make Prediction
    prediction = forest_regression.predict(features)

    # print(prediction.shape)
    st.header('Prediction of MEDV')
    st.info(round(prediction[0],4))

  

    


