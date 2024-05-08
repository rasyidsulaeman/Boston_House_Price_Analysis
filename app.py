# importing libraries

import streamlit as st
from boston import * 
import time
import shap
from sklearn.ensemble import RandomForestRegressor
from streamlit_shap import st_shap

# give description and title of the project
st.write("""
# Boston House Price Prediction
         
A web application to predict and analyze Boston House Price dataset.
Powered by **streamlit**.
         
For more information about the dataset, check [this website](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
""")
st.divider()

# use sidebar to layout-ing the website
sidebar = ('Exploratory Data Analysis', 'Machine Learning Boston House Prediction')
with st.sidebar:
    st.markdown('##### Author : Rasyid Sulaeman')
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
        time.sleep(5)
    st.success('Done!')

    st.write("""
    #### Outlier Detection
    
    Compute number of outlier for each attribute along with its percentage and display them using table; sorted from highest outlier to lowest.
    """)
    outlier = boston.outlier_calculation(df)
    st.dataframe(outlier)

elif radio == sidebar[1]:

    st.header('Specified Input parameters')
    
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

    # Build Regression Model
    model = RandomForestRegressor()
    model.fit(X, Y)

    # Apply Model to Make Prediction
    prediction = model.predict(features)

    # print(prediction.shape)
    st.header('Prediction of MEDV')
    st.metric('Prediction', round(prediction[0],3), label_visibility='hidden')

    # # Explaining the model's predictions using SHAP values
    # # https://github.com/slundberg/shap
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X)

    # st.header('Feature Importance')
    # plt.title('Feature importance based on SHAP values')
    # st_shap(shap.summary_plot(shap_values, X))

   





