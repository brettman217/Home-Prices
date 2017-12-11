
#Kaggle Home Price dataset fit with variouse regression techniques for high dimensionality data

# import all the necessaries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read in data
df = pd.read_csv('HomePrice_train.csv')

# Define y early on as the target (unscaled so far)
y=np.ravel(df.SalePrice)

#Drop columns with too many nan values to be useful
df.isnull().sum()
df = df.drop(['LotFrontage', 'Alley','FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'
              ,'Condition2', 'RoofMatl','Exterior1st', 'Exterior2nd', 'GarageYrBlt', 'Heating',
              'Utilities', 'GarageQual', 'HouseStyle', 'Electrical'], axis = 1)

# Fill nan in remaining datafraime in place using 'None' signifier to be picked up by dummies
# Garage qualitative fills
df['GarageType'].fillna('None', inplace = True) 
df['GarageFinish'].fillna('None', inplace = True)
df['GarageCond'].fillna('None', inplace = True)

# Basement qualitative fills
df['BsmtQual'].fillna('None', inplace = True)
df['BsmtCond'].fillna('None', inplace = True)   
df['BsmtExposure'].fillna('None', inplace = True)
df['BsmtFinType1'].fillna('None', inplace = True)

# Quantitative Fills for original dataframe training data
df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace = True)


#Encode all categoricals to dataframe (no further processing yet)
df = pd.get_dummies(df, drop_first = True)
X = df.drop(['Id', 'SalePrice'], axis =1)


#Scale all x and y features for use in ridge, lasso, elastic regressions:
from sklearn.preprocessing import StandardScaler
Xscaler = StandardScaler() # can also do on whole dataframe
yscaler = StandardScaler()
Xscl = pd.DataFrame(Xscaler.fit_transform(X), columns = X.columns)
yscl = yscaler.fit_transform(y.reshape(-1,1))
yscl = np.ravel(yscl)

########################################################################################################### Ridge
# Fitting Ridge regression
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(Xscl, yscl.reshape(-1,1))
Keep = ridge.coef_


######################################################################################################### Lasso
# Fitting Lasso regression
from sklearn.linear_model import LassoCV
lasso = LassoCV()
lasso.fit(Xscl, np.ravel(yscl))
Keep = pd.DataFrame(lasso.coef_, index = X.columns)


######################################################################################################### Elastic Net
# Elastic Net
from sklearn.linear_model import ElasticNetCV
elastic = ElasticNetCV()
elastic.fit(Xscl, np.ravel(yscl))


######################################################################################################### Test Data
df_test = pd.read_csv('HomePrice_test.csv')

# Define y early on as the target (unscaled so far)
df_test.isnull().sum()
df_test = df_test.drop(['LotFrontage', 'Alley','FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'
              ,'Condition2', 'RoofMatl','Exterior1st', 'Exterior2nd', 'GarageYrBlt', 'Heating',
              'Utilities', 'GarageQual', 'HouseStyle', 'Electrical'], axis = 1)

# Fill nan in remaining datafraime in place using 'None' signifier to be picked up by dummies
# Garage test data qualitative fills
df_test['GarageType'].fillna('None', inplace = True) 
df_test['GarageFinish'].fillna('None', inplace = True)
df_test['GarageCond'].fillna('None', inplace = True)

# Basement test data qualitative fills
df_test['BsmtQual'].fillna('None', inplace = True)
df_test['BsmtCond'].fillna('None', inplace = True)   
df_test['BsmtExposure'].fillna('None', inplace = True)
df_test['BsmtFinType1'].fillna('None', inplace = True)

# Fill remaining NA quantitative values with zeroes
#Basement quantitative fills test data
df_test['BsmtFinSF1'].fillna(0, inplace = True)
df_test['BsmtFinSF2'].fillna(0, inplace = True)
df_test['BsmtUnfSF'].fillna(0, inplace = True)
df_test['TotalBsmtSF'].fillna(0, inplace = True)
df_test['BsmtFullBath'].fillna(0, inplace = True)
df_test['BsmtHalfBath'].fillna(0, inplace = True)

#Garage quantitative fills test data
df_test['GarageCars'].fillna(0, inplace = True)
df_test['GarageArea'].fillna(0, inplace = True)

# Fill remaining test data quantitative NAs with averages where enough data exists
df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mean(), inplace = True)


#Encode all categoricals to testing dataframe
df_test = pd.get_dummies(df_test, drop_first = True)
X_test = df_test.drop(['Id'], axis =1)

#Scale test data X features for final fits
X_test_scl = pd.DataFrame(Xscaler.fit_transform(X_test), columns = X_test.columns)

############################################################################# FIT FINAL MODELS TO X_test scaled

# Final model predictions in scaled format
Lasso_Pred = lasso.predict(X_test_scl)
Ridge_Pred  = ridge.predict(X_test_scl)
Elastic_Pred = elastic.predict(X_test_scl)

# Final model predictions in original $ units format
Lasso_Final = yscaler.inverse_transform(Lasso_Pred)
Ridge_Final  = yscaler.inverse_transform(Ridge_Pred)
Elastic_Final = yscaler.inverse_transform(Elastic_Pred)













