import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier


from sklearn.model_selection import RandomizedSearchCV, cross_validate
import numpy as np
# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



attributes = pd.read_csv('datasets/machine_learning/scoutium_attributes.csv', sep=";")
potential_labels = pd.read_csv('datasets/machine_learning/scoutium_potential_labels.csv', sep=";")


df = pd.merge(attributes, potential_labels, on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'], how='left')

################################################
# 1. Exploratory Data Analysis
################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

check_df(df)


# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)



# Sayısal değişkenlerin incelenmesi
df[num_cols].describe().T
df.head

# for col in num_cols:
#     num_summary(df, col, plot=True)

# Sayısal değişkenkerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)

# Target ile sayısal değişkenlerin incelemesi
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))

df.head()

# Kalecileri bu sınıflandırmaya dahil etmek istemiyoruz.
# position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldıralım

df = df.loc[~(df["position_id"]==1)]



"""               potential_label      Ratio
average                   8497  79.189189
highlighted               2097  19.543336
below_average              136   1.267474
##########################################"""

# potentional_label sütununda below_average sınıfının yoğunluğu çok düşük.
# Ben bu verileri silmeyi tercih ediyorum.

df = df.loc[~(df["potential_label"] == "below_average")]
df["potential_label"].value_counts()

# Label'ımızı 2 sınıfa düşürdük.


# Bu tip bir veri setinde oyuncuyu satırda tekilleştirip, oyuncuya verilen tüm puanları sıralamalıyız
# Değişkenlerimiz scoutların İD'leri olacak. Pivot Table kullanıyorum

table1 = pd.pivot_table(df, values="attribute_value", index=["player_id", "position_id", "potential_label"], columns=["attribute_id"])
table1.head()

# Bir oyuncu birden fazla pozisyon için değerlendirilmiş olabilir. Bu yüzden groupby yapmadık.


# Veri setini daha düzgün bir biçime sokalım

table1 = table1.reset_index()
table1.head()


# İlerde sorun yaşamamak için değişken isimlerini "string" yapaım

table1.columns = table1.columns.astype(str)


# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(table1, cat_th=8, car_th=20)

table1.position_id.value_counts()

for col in cat_cols:
    cat_summary(df, col)

"""             potential_label      Ratio
average                 7922  80.068729
highlighted             1972  19.931271
##########################################
"""

# Hedef değişkenimiz potentional_label'i binary haline getirelim

labelencoder = LabelEncoder()
table1["potential_label"] = labelencoder.fit_transform(table1["potential_label"])
table1.head()



# Sayısal değişkenleri standartlaştırmamız gerekecek.

num_cols = [col for col in table1.columns if "player_id" not in col]
num_cols = num_cols[2:]
num_cols
column_type = df["potential_label"].dtype
# Hatalı bir işlem yaparsak kolayca geri dönebilelim diye kopya bıraktım.

df = table1.copy()

scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(scaled, columns=df[num_cols].columns)
df.head()
df.to_csv("/Users/mrpurtas/Desktop/df_pivot_table.csv", index=True)

df = pd.read_csv("/Users/mrpurtas/Desktop/df_pivot_table.csv")

for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


# Modellerimizi kuralım

y = df["potential_label"]
X = df.drop(["potential_label"], axis=1)




"""
# Hiperparametre aralıklarını genişletme
knn_params = {"n_neighbors": range(1, 101)}  # 1 ile 100 arasında

cart_params = {
    'max_depth': range(1, 51),  # 1 ile 50 arasında
    "min_samples_split": range(2, 51)  # 2 ile 50 arasında
}

rf_params = {
    "max_depth": [5, 10, 20, None],
    "max_features": [3, 5, 7, "sqrt", "log2"],  # 'auto' değeri çıkarıldı
    "min_samples_split": [2, 10, 15, 20, 30],
    "n_estimators": [100, 200, 300, 500]
}


xgboost_params = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 8, 10, 15],
    "n_estimators": [50, 100, 200, 300, 500],
    "colsample_bytree": [0.3, 0.5, 0.7, 1]
}

lightgbm_params = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 300, 500, 800],
    "colsample_bytree": [0.3, 0.5, 0.7, 1]
}


# AdaBoost için genişletilmiş hiperparametre aralıkları
adaboost_params = {
    "n_estimators": range(50, 501, 50),  # 50'den 500'e kadar, 50'şer artışla
    "learning_rate": np.linspace(0.001, 2, 20)  # 0.001'den 2'ye kadar lineer aralık
}

naive_bayes_params = {
    "var_smoothing": np.logspace(-9, -1, 20)  # 10^-9'dan 10^-1'e kadar logaritmik aralık
}
"""

knn_params = {
    'kneighborsclassifier__n_neighbors': np.arange(1, 100)  # 1'den 100'e kadar tamsayılar
}

cart_params = {
    'max_depth': np.arange(1, 51),  # 1 ile 50 arasında
    "min_samples_split": np.arange(2, 51)  # 2 ile 50 arasında
}

rf_params = {
    "max_depth": np.arange(10, 16),  # Focus around 13
    "max_features": [3, 4, 'sqrt', 6, 7],  # Numerical values around sqrt
    "min_samples_split": np.arange(12, 41),  # Higher values
    "n_estimators": np.arange(50, 151, 10)  # Lower range focusing around 60
}

xgboost_params = {
    "learning_rate": np.linspace(0.01, 0.2, 5),  # 0.01'den 0.2'ye kadar 5 değer
    "max_depth": np.arange(3, 16),  # 3 ile 15 arasında
    "n_estimators": np.arange(50, 501, 50),  # 50'den 500'e kadar, 50'şer adımla
    "colsample_bytree": np.linspace(0.3, 1, 4)  # 0.3'den 1'e kadar 4 değer
}

lightgbm_params = {
    "learning_rate": np.linspace(0.001, 0.2, 100),  # 0.01'den 0.2'ye kadar 5 değer
    "n_estimators": np.arange(500, 1001, 50),  # 100'den 800'e kadar, 100'er adımla
    "colsample_bytree": np.linspace(0.3, 5, 20)  # 0.3'den 1'e kadar 4 değer
}

adaboost_params = {
    "n_estimators": np.arange(50, 501, 50),  # 100'den 500'e kadar, 50'şer artışla
    "learning_rate": np.linspace(0.001, 0.1, 10)  # 0.001'den 0.1'e kadar lineer aralık
}

naive_bayes_params = {
    'gaussiannb__var_smoothing': np.logspace(0,-9, num=50)
}

lr_params = {
    "penalty": ['l1', 'l2', 'elasticnet', 'none'],
    "C": np.logspace(-4, 4, 20),
    "solver": ['newton-cg', 'liblinear', 'sag', 'saga'],
    "max_iter": np.arange(100, 1001, 100)
}

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Sınıflandırıcıları ve parametrelerini bir listeye ekleyin
knn_pipeline = make_pipeline(SMOTE(), KNeighborsClassifier())
nb_smote_pipeline = make_pipeline(SMOTE(), GaussianNB())

classifiers = [
    ('Logistic Regression', LogisticRegression(class_weight='balanced', verbose=0), lr_params),
    ('KNN with SMOTE', make_pipeline(SMOTE(), KNeighborsClassifier()), knn_params),
    ("CART", DecisionTreeClassifier(class_weight='balanced'), cart_params),
    ("RF", RandomForestClassifier(class_weight='balanced'), rf_params),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=sum(y==0)/sum(y==1)), xgboost_params),
    ('LightGBM', LGBMClassifier(is_unbalance=True, verbose=-1), lightgbm_params),
    ('AdaBoost', AdaBoostClassifier(), adaboost_params)
]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc", n_iter=100):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        rs_best = RandomizedSearchCV(classifier, params, cv=cv, n_iter=n_iter, n_jobs=-1, verbose=0, random_state=42).fit(X, y)
        final_model = classifier.set_params(**rs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {rs_best.best_params_}", end="\n\n")
        best_models[name] = final_model

    return best_models

best_models = hyperparameter_optimization(X, y, cv=3, scoring="roc_auc")

"""ROC AUC (Receiver Operating Characteristic Area Under Curve): Modelin sınıfları ayırt etme 
yeteneğini değerlendirir. ROC AUC, modelin farklı eşik değerlerindeki performansını bir bütün 
olarak ölçer ve dengesiz veri setlerinde özellikle yararlıdır."""

"""Hyperparameter Optimization....
########## Logistic Regression ##########
roc_auc (Before): 0.5551
roc_auc (After): 0.7998
Logistic Regression best params: {'solver': 'newton-cg', 'penalty': 'l2', 'max_iter': 1000, 'C': 1.623776739188721}
########## KNN ##########
roc_auc (Before): 0.5022
roc_auc (After): 0.5022
KNN best params: {'n_neighbors': 96}
########## CART ##########
roc_auc (Before): 0.6611
roc_auc (After): 0.743
CART best params: {'min_samples_split': 46, 'max_depth': 2}
########## RF ##########
roc_auc (Before): 0.8797
roc_auc (After): 0.8852
RF best params: {'n_estimators': 150, 'min_samples_split': 20, 'max_features': 4, 'max_depth': 14}
########## XGBoost ##########
roc_auc (Before): 0.8308
roc_auc (After): 0.8804
XGBoost best params: {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.0575, 'colsample_bytree': 0.5333333333333333}
########## LightGBM ##########
roc_auc (Before): 0.8493
roc_auc (After): 0.8365
LightGBM best params: {'n_estimators': 850, 'learning_rate': 0.00301010101010101, 'colsample_bytree': 1.0}
########## Naive Bayes ##########
roc_auc (Before): 0.623
roc_auc (After): 0.623
Naive Bayes best params: {'var_smoothing': 1e-09}
########## AdaBoost ##########
roc_auc (Before): 0.7788
roc_auc (After): 0.7302
AdaBoost best params: {'n_estimators': 500, 'learning_rate': 0.001}"""

"""
best_models
'Logistic Regression': LogisticRegression(C=1.623776739188721, max_iter=1000, solver='newton-cg'),
 'KNN': KNeighborsClassifier(n_neighbors=96),
 'CART': DecisionTreeClassifier(max_depth=2, min_samples_split=46),
 'RF': RandomForestClassifier(max_depth=14, max_features=4, min_samples_split=20,
                        n_estimators=150),
 'XGBoost': XGBClassifier(base_score=None, booster=None, callbacks=None,
               colsample_bylevel=None, colsample_bynode=None,
               colsample_bytree=0.5333333333333333, device=None,
               early_stopping_rounds=None, enable_categorical=False,
               eval_metric='logloss', feature_types=None, gamma=None,
               grow_policy=None, importance_type=None,
               interaction_constraints=None, learning_rate=0.0575, max_bin=None,
               max_cat_threshold=None, max_cat_to_onehot=None,
               max_delta_step=None, max_depth=4, max_leaves=None,
               min_child_weight=None, missing=nan, monotone_constraints=None,
               multi_strategy=None, n_estimators=50, n_jobs=None,
               num_parallel_tree=None, random_state=None, ...),
 'LightGBM': LGBMClassifier(colsample_bytree=0.5333333333333333,
                learning_rate=0.05125252525252525, verbose=-1),
 'Naive Bayes': GaussianNB(var_smoothing=1.0),
 'AdaBoost': AdaBoostClassifier(learning_rate=0.001, n_estimators=500)}"""


Hyperparameter Optimization....
########## Logistic Regression ##########
roc_auc (Before): 0.3614
roc_auc (After): 0.8272
Logistic Regression best params: {'solver': 'newton-cg', 'penalty': 'l2', 'max_iter': 600, 'C': 0.23357214690901212}
########## KNN with SMOTE ##########
roc_auc (Before): 0.5
roc_auc (After): 0.4907
KNN with SMOTE best params: {'kneighborsclassifier__n_neighbors': 70}
########## CART ##########
roc_auc (Before): 0.697
roc_auc (After): 0.7236
CART best params: {'min_samples_split': 46, 'max_depth': 1}
########## RF ##########
roc_auc (Before): 0.8778
roc_auc (After): 0.8786
RF best params: {'n_estimators': 140, 'min_samples_split': 18, 'max_features': 3, 'max_depth': 14}
########## XGBoost ##########
roc_auc (Before): 0.8112
roc_auc (After): 0.8676
XGBoost best params: {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.15250000000000002, 'colsample_bytree': 0.3}
########## LightGBM ##########
roc_auc (Before): 0.8506
roc_auc (After): 0.8634
LightGBM best params: {'n_estimators': 850, 'learning_rate': 0.14572727272727273, 'colsample_bytree': 0.5473684210526316}
########## AdaBoost ##########
roc_auc (Before): 0.7788
roc_auc (After): 0.7877
AdaBoost best params: {'n_estimators': 100, 'learning_rate': 0.023000000000000003}

""""
best_modelss = hyperparameter_optimization(X, y, cv=3, scoring="f1_macro")

Hyperparameter Optimization....
########## Logistic Regression ##########
f1_macro (Before): 0.7406
f1_macro (After): 0.7734
Logistic Regression best params: {'solver': 'newton-cg', 'penalty': 'l2', 'max_iter': 600, 'C': 0.23357214690901212}
########## KNN with SMOTE ##########
f1_macro (Before): 0.3527
f1_macro (After): 0.3527
KNN with SMOTE best params: {'kneighborsclassifier__n_neighbors': 59}
########## CART ##########
f1_macro (Before): 0.7597
f1_macro (After): 0.7597
CART best params: {'min_samples_split': 46, 'max_depth': 1}
########## RF ##########
f1_macro (Before): 0.7558
f1_macro (After): 0.7708
RF best params: {'n_estimators': 80, 'min_samples_split': 14, 'max_features': 3, 'max_depth': 15}
########## XGBoost ##########
f1_macro (Before): 0.7955
f1_macro (After): 0.7955
XGBoost best params: {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.0575, 'colsample_bytree': 0.5333333333333333}
########## LightGBM ##########
f1_macro (Before): 0.7722
f1_macro (After): 0.7454
LightGBM best params: {'n_estimators': 850, 'learning_rate': 0.14572727272727273, 'colsample_bytree': 0.5473684210526316}
########## AdaBoost ##########
f1_macro (Before): 0.7667
f1_macro (After): 0.7667
AdaBoost best params: {'n_estimators': 100, 'learning_rate': 0.023000000000000003}"""


"""best_modelss f1_macro ıcın en ıyı bulduugm parametreler
{'Logistic Regression': LogisticRegression(C=0.23357214690901212, 'penalty'='l2', class_weight='balanced', max_iter=600,
                    solver='newton-cg'),
 'KNN with SMOTE': Pipeline(steps=[('smote', SMOTE()),
                 ('kneighborsclassifier', KNeighborsClassifier(n_neighbors=59))]),
 'CART': DecisionTreeClassifier(class_weight='balanced', max_depth=1,
                        min_samples_split=46),
 'RF': RandomForestClassifier(class_weight='balanced', 'n_estimators'= 80, 'min_samples_split'= 14, 'max_features'= 3, 'max_depth'= 15),
 'XGBoost': XGBClassifier(base_score=None, booster=None, callbacks=None,
               colsample_bylevel=None, colsample_bynode=None,
               colsample_bytree=0.3, device=None, early_stopping_rounds=None,
               enable_categorical=False, eval_metric='logloss',
               feature_types=None, gamma=None, grow_policy=None,
               importance_type=None, interaction_constraints=None,
               learning_rate=0.15250000000000002, max_bin=None,
               max_cat_threshold=None, max_cat_to_onehot=None,
               max_delta_step=None, max_depth=4, max_leaves=None,
               min_child_weight=None, missing=nan, monotone_constraints=None,
               multi_strategy=None, n_estimators=850, n_jobs=None,
               num_parallel_tree=None, random_state=None, ...),
 'LightGBM': LGBMClassifier(colsample_bytree=0.5473684210526316, is_unbalance=True,
                learning_rate=0.14572727272727273, n_estimators=850, verbose=-1),
 'AdaBoost': AdaBoostClassifier(learning_rate=0.023000000000000003, n_estimators=100)}"""


"""
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    # Oylama sınıflandırıcısını oluşturma
    voting_clf = VotingClassifier(estimators=[
        ('Logistic Regression', best_models["Logistic Regression"]),
        ('CART', best_models["CART"]),
        ('RF', best_models["RF"]),
        ('XGBoost', best_models["XGBoost"]),
        ('LightGBM', best_models["LightGBM"]),
        ('AdaBoost', best_models["AdaBoost"])
        ],
        voting='soft'
    ).fit(X, y)

    # Modelin çapraz doğrulama sonuçlarını hesaplama
    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1_macro", "roc_auc"], n_jobs=-1)

    # Performans metriklerini yazdırma
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1 Macro Score: {cv_results['test_f1_macro'].mean()}")
    print(f"ROC AUC: {cv_results['test_roc_auc'].mean()}")

    return voting_clf

voting_classifier(best_models, X, y)

Accuracy: 0.8635690235690235
F1 Macro Score: 0.7690536731039552
ROC AUC: 0.8940451021846372
"""

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    # Oylama sınıflandırıcısını oluşturma
    voting_clf = VotingClassifier(estimators=[
        ('Logistic Regression', best_models["Logistic Regression"]),
        ('CART', best_models["CART"]),
        ('RF', best_models["RF"]),
        ('XGBoost', best_models["XGBoost"]),
        ('LightGBM', best_models["LightGBM"]),
        ],
        voting='soft'
    ).fit(X, y)

    # Modelin çapraz doğrulama sonuçlarını hesaplama
    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1_macro", "roc_auc"], n_jobs=-1)

    # Performans metriklerini yazdırma
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1 Macro Score: {cv_results['test_f1_macro'].mean()}")
    print(f"ROC AUC: {cv_results['test_roc_auc'].mean()}")

    return voting_clf

voting_classifier(best_models, X, y)

"""Accuracy: 0.8672053872053873
F1 Macro Score: 0.7812132088727102
ROC AUC: 0.8958773784355178"""



X.columns
random_user = X.sample(1, random_state=45)
voting_clf.predict(random_user)

joblib.dump(voting_clf, "/Users/mrpurtas/Desktop/voting_clf2.pkl")

new_model = joblib.load("datasets/voting_clf2.pkl")
new_model.predict(random_user)

