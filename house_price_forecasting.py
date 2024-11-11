import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor


#Downloading Data Set
headers  = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('housing.csv', sep='\s+', names= headers)
print(df.head())

#-------------------------------DATA PREPROCESSING---------------------------------
#Finding Empty Cells in the Data Set
print("Shape of Dataset:", df.shape)
print(df.isnull().sum())

#To learn the data types of the data in each column in the data set
print(df.info())

#To receive statistical information about our data set
print(df.describe().T)

#To detect the presence of anomalous data in each column in the data set
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()
for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# Cleaning the target column MEDV(HousePrices) Outliers Data
df = df[~(df['MEDV'] >= 50.0)]


#Data set correlation calculation
plt.figure(figsize=(20,10))
sns.heatmap(df.corr().abs(), annot=True)

#Normalisation of Data
scaler=MinMaxScaler(feature_range=(0,1))
column_sels  = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df=pd.DataFrame(scaler.fit_transform(df), columns=column_sels)


#-------------------------------FEATURE SELECTION------------------------------
"""From the correlation matrix, we see that TAX and RAD are highly correlated. 
Therefore there is no need to take the RAD column LSTAT, INDUS, RM, TAX, NOX, 
PTRAIO columns have a high correlation score with MEDV, which is a good indicator 
to be used as a predictor."""

# Separation of Dependent and Independent Variables

X=df.drop(['CRIM', 'ZN', 'CHAS', 'RAD', 'B', 'MEDV'], axis=1)
y = pd.DataFrame(df['MEDV'])


#------------------------------MODELLING-------------------------------------

#Train,test split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 4)

#Fitting the training set according to Decision Tree Regression
reg =DecisionTreeRegressor(max_depth = 4, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

#--------------------------CALCULATION OF ERROR METRICS---------------------

print("MSE=%0.4f"%mean_squared_error(y_test, y_pred))
print("R^2 Score=%0.2f"%r2_score(y_test, y_pred)) 
