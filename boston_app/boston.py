import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

class BostonEDA():

    def __init__(self, path):
        self.path = path
        self.plot_option = ('Histogram', 'Regression-Scatter', 'Box')

    def data_loaded(self):
        df = pd.read_csv(self.path)
        df.columns = map(str.upper, df.columns)
        return df
    
    def heatmap(self, df):
        fig, ax = plt.subplots(figsize=(14,7))
        sns.heatmap(df.corr(numeric_only = True),cbar=True,annot=True, ax=ax)
        return fig
    
    def outlier_calculation(self, df):
        outliers_list = []

        for col in df.columns:
            percentile25 = df[col].quantile(0.25)
            percentile75 = df[col].quantile(0.75)
            
            IQR  = percentile75 - percentile25
            
            upper_limit = percentile75 + 1.5*IQR
            lower_limit = percentile25 - 1.5*IQR
            
            outliers = df[(df[col] > upper_limit) | (df[col] < lower_limit)]
            percentage = outliers.shape[0] / df.shape[0] * 100

            outlier_dict = {'Feature' : col.upper(),
                            'Outlier Count' : outliers.shape[0],
                            'Outlier Percentage' : f'{percentage:.2f}' + '%',
                            'Max' : df[col].max(),
                            'Min' : df[col].min()}
            
            outliers_list.append(outlier_dict)

        return pd.DataFrame(outliers_list).sort_values(by='Outlier Count', ascending=False)

    def plots(self, df, option):

        fig, axes = plt.subplots(2,7, figsize=(20,14)) 
        if option == self.plot_option[0]:

            # histplot to display normallity
            for index, axs in enumerate(axes.flatten()):
                plot = sns.histplot(data=df, x=df.columns[index], kde=True, ax=axs)
                plot.set(xlabel='', ylabel='Frequency', title=df.columns[index].upper())
            
        elif option == self.plot_option[1]:

            # regplot for linear relationship
            features_cols = df.columns[:-1]
            for index, axs in enumerate(axes.flatten()[:len(features_cols)]):
                plot = sns.regplot(x=features_cols[index], y='MEDV', data=df, ax=axs)
                plot.set(xlabel='', ylabel='Median value of Homes', title=features_cols[index].upper())
                
        elif option == self.plot_option[2]:

            # box plot for outlier
            for index, axs in enumerate(axes.flatten()):
                plot = sns.boxplot(data=df, y=df.columns[index], ax=axs)
                plot.set(xlabel='', ylabel='', title=df.columns[index])

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        return fig
