import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
#import graphviz

#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("hitters.csv")
#######################################
# ANALYSİS OF CATEGORICAL VARIABLES
#######################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, cat_but_car, num_cols


cat_cols, cat_but_car, num_cols = grab_col_names(df)


######################################
# OUTLIER ANALYSIS
######################################
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quartile1 = dataframe[variable].quantile(low_quantile)
    quartile3 = dataframe[variable].quantile(up_quantile)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


#for col in num_cols:
#    print(col, check_outlier(df, col))
#CHits True
#CHmRun True
#CWalks True

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)


#for col in num_cols:
#    print(col, check_outlier(df, col))


######################################
# MISSING VALUE ANALYSIS
######################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


#print(missing_values_table(df, True)) #['Salary']
#print(df["Salary"])
df = df.dropna()
#print(df["Salary"])


######################################
# Label Encoding & One-Hot Encoding
######################################
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)


#############################################
# Base Models
#############################################
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
y = df["Salary"]
X = df.drop(["Salary"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
log_model = LinearRegression().fit(X, y)
y_pred = log_model.predict(X)

#print(mean_squared_error(y, y_pred)) #mse = 90820.74696609698
#print(log_model.score(X, y)) #score = 0.5520207444792639


#####################
# CV ile Başarı Değerlendirme
#####################
linear_model = LinearRegression()

#print(np.mean(cross_val_score(linear_model, X, y, cv=5))) #CV = 0.3504639422535659


#############################################
# FEATURE EXTRACTİON
#############################################
new_num_cols = [col for col in num_cols if col != "Salary"]
df[new_num_cols] = df[new_num_cols]+0.0000000001
print(df[new_num_cols])
df["NEW_Hits"] = df["Hits"] / df["CHits"] + df["Hits"]
df["NEW_RBI"] = df["RBI"] / df["CRBI"]
df["NEW_Walks"] = df["Walks"] / df["CWalks"]
df["NEW_PutOuts"] = df["PutOuts"] * df["Years"]
df["Hits_Success"] = (df["Hits"] / df["AtBat"]) * 100
df["NEW_CRBI*CATBAT"] = df["CRBI"] * df["CAtBat"]
df["NEW_RBI"] = df["RBI"] / df["CRBI"]
df["NEW_Chits"] = df["CHits"] / df["Years"]
df["NEW_CHmRun"] = df["CHmRun"] * df["Years"]
df["NEW_CRuns"] = df["CRuns"] / df["Years"]
df["NEW_Chits"] = df["CHits"] * df["Years"]
df["NEW_RW"] = df["RBI"] * df["Walks"]
df["NEW_RBWALK"] = df["RBI"] / df["Walks"]
df["NEW_CH_CB"] = df["CHits"] / df["CAtBat"]
df["NEW_CHm_CAT"] = df["CHmRun"] / df["CAtBat"]
df["NEW_Diff_Atbat"] = df["AtBat"] - (df["CAtBat"] / df["Years"])
df["NEW_Diff_Hits"] = df["Hits"] - (df["CHits"] / df["Years"])
df["NEW_Diff_HmRun"] = df["HmRun"] - (df["CHmRun"] / df["Years"])
df["NEW_Diff_Runs"] = df["Runs"] - (df["CRuns"] / df["Years"])
df["NEW_Diff_RBI"] = df["RBI"] - (df["CRBI"] / df["Years"])
df["NEW_Diff_Walks"] = df["Walks"] - (df["CWalks"] / df["Years"])


df = pd.get_dummies(df)


y = df["Salary"]
X = df.drop(["Salary"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
log_model = LinearRegression().fit(X, y)
y_pred = log_model.predict(X)

#print(mean_squared_error(y, y_pred)) #mse = 43176.28297771722
#print(log_model.score(X, y)) #score = 0.7870301693099847

linear_model = LinearRegression()

#print(np.mean(cross_val_score(linear_model, X, y, cv=5))) #CV = 0.5423302014460163

