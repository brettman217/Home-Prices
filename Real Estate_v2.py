

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('HomePrice_train.csv')
sns.lmplot(x='Id', y='SalePrice', data = df)

#Split X and y
y=df.SalePrice


#Drop too many na columns
df.isnull().sum()
df = df.drop(['LotFrontage', 'Alley','FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature' ], axis = 1)

# Fill garages nan with None
df['GarageType'].fillna('None', inplace = True)
df['GarageYrBlt'].fillna('None', inplace = True)   
df['GarageFinish'].fillna('None', inplace = True)
df['GarageQual'].fillna('None', inplace = True)
df['GarageCond'].fillna('None', inplace = True)
df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace = True)

#Fill basement nan with None
df['BsmtQual'].fillna('None', inplace = True)
df['BsmtCond'].fillna('None', inplace = True)   
df['BsmtExposure'].fillna('None', inplace = True)
df['BsmtFinType1'].fillna('None', inplace = True)


#Encode all categoricals
df = pd.get_dummies(df, drop_first = True)
X = df.drop('SalePrice', axis =1)

#Check correlation to see what else can be dropped
correlation_check = df.corr()
Sale_Corr = correlation_check.SalePrice

#Find variables with greater than 30% correlation to sale price and filter - drop sale price from itself
Start_Vars = Sale_Corr[Sale_Corr > .3]
Start_Vars = Start_Vars.drop('SalePrice')
Start_Vars.index

#redefine X to variables with corr> 0.3
X = df[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath',
       'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt',
       'Exterior1st_VinylSd', 'Exterior2nd_VinylSd', 'MasVnrType_Stone',
       'ExterQual_Gd', 'Foundation_PConc', 'BsmtExposure_Gd',
       'BsmtFinType1_GLQ', 'KitchenQual_Gd', 'GarageType_Attchd',
       'SaleType_New', 'SaleCondition_Partial']]



#Get model
import statsmodels.formula.api as sm
import statsmodels.tools as smtools
X = smtools.add_constant(data = X, prepend = True)

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



##########################################################################################################
# Fit and eliminate
#Something Wrong with MaxVnrArea
#Use list from X to get back full list, 'TotRmsAbvGrd', 'TotRmsAbvGrd', 'GarageCars',
#'Exterior1st_VinylSd', 'Exterior2nd_VinylSd', 'MasVnrType_Stone', 'Foundation_PConc', , 'SaleCondition_Partial'
#'ExterQual_Gd', 'OpenPorchSF', 'GarageType_Attchd', , '2ndFlrSF', 'BsmtFinType1_GLQ', '1stFlrSF', 'KitchenQual_Gd',
X_Opt = X_train[['const','OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'TotalBsmtSF', 'GrLivArea',
       'Fireplaces', 'GarageArea', 'WoodDeckSF',
       'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'BsmtExposure_Gd',
       'SaleType_New']]
regressor_OLS = sm.OLS(endog = y_train, exog = X_Opt).fit()
regressor_OLS.summary()

#Align X_Test to significant variables only
SigOnly_X_test = X_test[['const','OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'TotalBsmtSF', 'GrLivArea',
       'Fireplaces', 'GarageArea', 'WoodDeckSF',
       'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'BsmtExposure_Gd',
       'SaleType_New']]

#Create open datafram to track results of each model
accuracy_list = pd.DataFrame(columns = ['model', 'accuracy', 'stdev'])

############################################################################################ FIT LR Model X_Opt Variables
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_Opt, y_train)
y_pred = lr.predict(SigOnly_X_test)

#check the outputs of model
from sklearn import metrics
intercept = lr.intercept_
lr.coef_
metrics.r2_score(y_test, y_pred) # r squared

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lr, X = X_Opt, y = y_train, cv = 10) #cv parameter is # folds
acc_mean = accuracies.mean() # average across the 10 folds
acc_stdev = accuracies.std() # variation across the 10 folds
accuracy_list= accuracy_list.append({'model': 'lr', 'accuracy': acc_mean, 'stdev': acc_stdev}, ignore_index=True)


########################################################################################## RANDOM FOREST 1000 to all X
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=1000)

rfr.fit(X_train, y_train)
rfcpred = rfr.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rfr, X = X_train, y = y_train, cv = 10) #cv parameter is # folds
acc_mean = accuracies.mean() # average across the 10 folds
acc_stdev = accuracies.std() # variation across the 10 folds
accuracy_list.append(['rfr', acc_mean,acc_stdev])
accuracy_list= accuracy_list.append({'model': 'rfr', 'accuracy': acc_mean, 'stdev': acc_stdev}, ignore_index=True)

accuracy_list.sort_values('accuracy', ascending = 0, inplace = True)


################################################################################################################## BRING IN COMPETITION DATA
import pandas as pd

comp = pd.read_csv('HomePrice_test.csv')

#Drop too many na columns aligned with train data
comp.isnull().sum()
comp = comp.drop(['LotFrontage', 'Alley','FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature' ], axis = 1)

# Fill garages nan and other Nan with None
comp['GarageType'].fillna('None', inplace = True)
comp['GarageYrBlt'].fillna('None', inplace = True)   
comp['GarageFinish'].fillna('None', inplace = True)
comp['GarageQual'].fillna('None', inplace = True)
comp['GarageCond'].fillna('None', inplace = True)

#Fill empty values with mean of column
comp['MasVnrArea'].fillna(comp['MasVnrArea'].mean(), inplace = True)

#Fill test set NAs with zeroes
comp['BsmtFinSF1'].fillna(0, inplace = True)   
comp['TotalBsmtSF'].fillna(0, inplace = True)   
comp['GarageArea'].fillna(0, inplace = True)  

#Fill basement nan with None
comp['BsmtQual'].fillna('None', inplace = True)
comp['BsmtCond'].fillna('None', inplace = True)   
comp['BsmtExposure'].fillna('None', inplace = True)
comp['BsmtFinType1'].fillna('None', inplace = True)


#Encode all categoricals
comp = pd.get_dummies(comp, drop_first = True)



#redefine X to variables with corr> 0.3 in exploration phase
Xcomp = comp[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath',
       'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt',
       'Exterior1st_VinylSd', 'Exterior2nd_VinylSd', 'MasVnrType_Stone',
       'ExterQual_Gd', 'Foundation_PConc', 'BsmtExposure_Gd',
       'BsmtFinType1_GLQ', 'KitchenQual_Gd', 'GarageType_Attchd',
       'SaleType_New', 'SaleCondition_Partial']]



#Add constant to frame
import statsmodels.formula.api as sm
import statsmodels.tools as smtools
Xcomp = smtools.add_constant(data = Xcomp, prepend = True)


##########################################################################################################
# Keep only significant columns from exploration phase
X_Opt_Comp = Xcomp[['const','OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'TotalBsmtSF', 'GrLivArea',
       'Fireplaces', 'GarageArea', 'WoodDeckSF',
       'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'BsmtExposure_Gd',
       'SaleType_New']]



############################################################################################ FIT FINAL MODELS
comp_lr = lr.predict(X_Opt_Comp)
comp_rfr = rfr.predict(X_Opt_Comp)





