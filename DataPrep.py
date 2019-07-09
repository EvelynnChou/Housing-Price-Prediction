#!/usr/bin/env python
# coding: utf-8


import csv
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import json



with open('D:/Users/p10260458/Desktop/玉山房價預測/新增資料夾/data/train_New_0612.csv', newline='') as csvFile:
    rows = csv.reader(csvFile, delimiter=',')
    
    for row in rows: 
        print(row)


train = pd.read_csv('D:/Users/p10260458/Desktop/玉山房價預測/新增資料夾/data/train_New_0612.csv',encoding='big5')
#train = train.set_index('building_id')
train.head()


train[["building_material","city","building_type","building_use","parking_way","town","village"]] = train[["building_material","city","building_type","building_use","parking_way","town","village"]].astype('category')

train=train.drop(columns=['txn_dt_Excel', 'txn_dt_SAS','building_complete_dt_Excel','building_complete_dt_SAS','unit_price'])
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
# print(train.dtypes)

train.head()

test = pd.read_csv('D:/Users/p10260458/Desktop/玉山房價預測/新增資料夾/data/test_New_0617.csv',encoding='big5')
#test = test.set_index('building_id')
test.head()


test[["building_material","city","building_type","building_use","parking_way","town","village"]] = test[["building_material","city","building_type","building_use","parking_way","town","village"]].astype('category')

test = test.drop(columns=['txn_dt_Excel', 'txn_dt_SAS','building_complete_dt_Excel','building_complete_dt_SAS'])

# print(test.dtypes)


train.isnull().sum()


train.fillna(train[['parking_area','parking_price','txn_floor','village_income_median']].mean().to_dict(),inplace = True)


train.isnull().sum()


train[['parking_area','parking_price','txn_floor','village_income_median']]


pk_a_mean = train['parking_area'].mean()
pk_p_mean = train['parking_price'].mean()
txn_f_mean = train['txn_floor'].mean()
vim_mean = train['village_income_median'].mean()
print([pk_a_mean, pk_p_mean, txn_f_mean,vim_mean])


test['parking_area'].fillna(train['parking_area'].mean(),inplace = True)
test['parking_price'].fillna(train['parking_price'].mean(),inplace = True)
test['txn_floor'].fillna(train['txn_floor'].mean(),inplace = True)
test['village_income_median'].fillna(train['village_income_median'].mean(),inplace = True)


test[['parking_area','parking_price','txn_floor','village_income_median']]


print(train.columns)


train_minMAX = train.copy()


col_names = ['txn_dt', 'total_floor', 'building_complete_dt', 'building_Age', 'parking_area', 'parking_price', 'txn_floor', 'land_area', 'building_area', 'lat', 'lon', 'village_income_median', 'town_population', 'town_area', 'town_population_density', 'doc_rate', 'master_rate', 'bachelor_rate', 'jobschool_rate', 'highschool_rate', 'junior_rate', 'elementary_rate', 'born_rate', 'death_rate', 'marriage_rate', 'divorce_rate',  'N_50', 'N_500', 'N_1000', 'N_5000', 'N_10000', 'I_10', 'I_50', 'I_index_50', 'I_100', 'I_250', 'I_500', 'I_index_500', 'I_1000', 'I_index_1000', 'I_5000', 'I_index_5000', 'I_10000', 'I_index_10000', 'I_MIN', 'II_10', 'II_50', 'II_index_50', 'II_100', 'II_250', 'II_500', 'II_index_500', 'II_1000', 'II_index_1000', 'II_5000', 'II_index_5000', 'II_10000', 'II_index_10000', 'II_MIN', 'III_10', 'III_50', 'III_index_50', 'III_100', 'III_250', 'III_500', 'III_index_500', 'III_1000', 'III_index_1000', 'III_5000', 'III_index_5000', 'III_10000', 'III_index_10000', 'III_MIN', 'IV_10', 'IV_50', 'IV_index_50', 'IV_100', 'IV_250', 'IV_500', 'IV_index_500', 'IV_1000', 'IV_index_1000', 'IV_5000', 'IV_index_5000', 'IV_10000', 'IV_index_10000', 'IV_MIN', 'V_10', 'V_50', 'V_index_50', 'V_100', 'V_250', 'V_500', 'V_index_500', 'V_1000', 'V_index_1000', 'V_5000', 'V_index_5000', 'V_10000', 'V_index_10000', 'V_MIN', 'VI_10', 'VI_50', 'VI_index_50', 'VI_100', 'VI_250', 'VI_500', 'VI_index_500', 'VI_1000', 'VI_index_1000', 'VI_5000', 'VI_index_5000', 'VI_10000', 'VI_index_10000', 'VI_MIN', 'VII_10', 'VII_50', 'VII_index_50', 'VII_100', 'VII_250', 'VII_500', 'VII_index_500', 'VII_1000', 'VII_index_1000', 'VII_5000', 'VII_index_5000', 'VII_10000', 'VII_index_10000', 'VII_MIN', 'VIII_10', 'VIII_50', 'VIII_index_50', 'VIII_100', 'VIII_250', 'VIII_500', 'VIII_index_500', 'VIII_1000', 'VIII_index_1000', 'VIII_5000', 'VIII_index_5000', 'VIII_10000', 'VIII_index_10000', 'VIII_MIN', 'IX_10', 'IX_50', 'IX_index_50', 'IX_100', 'IX_250', 'IX_500', 'IX_index_500', 'IX_1000', 'IX_index_1000', 'IX_5000', 'IX_index_5000', 'IX_10000', 'IX_index_10000', 'IX_MIN', 'X_10', 'X_50', 'X_index_50', 'X_100', 'X_250', 'X_500', 'X_index_500', 'X_1000', 'X_index_1000', 'X_5000', 'X_index_5000', 'X_10000', 'X_index_10000', 'X_MIN', 'XI_10', 'XI_50', 'XI_index_50', 'XI_100', 'XI_250', 'XI_500', 'XI_index_500', 'XI_1000', 'XI_index_1000', 'XI_5000', 'XI_index_5000', 'XI_10000', 'XI_index_10000', 'XI_MIN', 'XII_10', 'XII_50', 'XII_index_50', 'XII_100', 'XII_250', 'XII_500', 'XII_index_500', 'XII_1000', 'XII_index_1000', 'XII_5000', 'XII_index_5000', 'XII_10000', 'XII_index_10000', 'XII_MIN', 'XIII_10', 'XIII_50', 'XIII_index_50', 'XIII_100', 'XIII_250', 'XIII_500', 'XIII_index_500', 'XIII_1000', 'XIII_index_1000', 'XIII_5000', 'XIII_index_5000', 'XIII_10000', 'XIII_index_10000', 'XIII_MIN', 'XIV_10', 'XIV_50', 'XIV_index_50', 'XIV_100', 'XIV_250', 'XIV_500', 'XIV_index_500', 'XIV_1000', 'XIV_index_1000', 'XIV_5000', 'XIV_index_5000', 'XIV_10000', 'XIV_index_10000', 'XIV_MIN']
train_features = train_minMAX[col_names]
scaler = pp.MinMaxScaler().fit(train_features.values)
train_features = scaler.transform(train_features.values)


train_minMAX[col_names] = train_features
#print(train_minMAX)


train_minMAX.head()


test_minMAX = test.copy()


test_features = test_minMAX[col_names]
test_features = scaler.transform(test_features.values)

test_minMAX[col_names] = test_features
#print(test_minMAX)


test_minMAX.head()



#normalizer = pp.Normalizer().fit(train)
#train_Norm = normalizer.transform(train)
#test_Norm = normalizer.transform(test)
#print(train_Norm)


#X_imputed_df = pd.DataFrame(train_Norm, columns = train.columns)
#X_imputed_df.head()


train_minMAX['trainornot'] = 1
train_minMAX.head()


# print(train_minMAX.dtypes)


test_minMAX['trainornot'] = 0
test_minMAX.head()


# print(test_minMAX.dtypes)


combine_minMax = pd.concat([train_minMAX,test_minMAX],sort=False)
# print(combine_minMax)


# print(combine_minMax.dtypes)


combine_minMax['town_uniq'] = combine_minMax['city'].astype(str) + '_' + combine_minMax['town'].astype(str)


combine_minMax['village_uniq'] = combine_minMax['city'].astype(str) + '_' + combine_minMax['town'].astype(str) + '_' + combine_minMax['village'].astype(str)


combine_minMax


# print(combine_minMax.dtypes)


combine_minMax[["building_material","town_uniq","village_uniq"]] = combine_minMax[["building_material","town_uniq","village_uniq"]].astype('category')

combine_minMax = combine_minMax.drop(columns=['town', 'village'])

# print(combine_minMax.dtypes)


train_minMax = combine_minMax[combine_minMax.trainornot == 1]


train_minMax = train_minMax.drop(columns=['trainornot'])

test_minMax = combine_minMax[combine_minMax.trainornot == 1]


test_minMax = test_minMax.drop(columns=['trainornot'])


train_minMax.to_csv('D:/Users/p10260458/Desktop/玉山房價預測/新增資料夾/data/train_minMax.csv', encoding='utf-8', index=False)

test_minMax.to_csv('D:/Users/p10260458/Desktop/玉山房價預測/新增資料夾/data/test_minMax.csv', encoding='utf-8', index=False)



combine_OneHot = pd.get_dummies(data=combine_minMax, columns=['city', 'building_material','building_type','building_use','parking_way','town_uniq','village_uniq'])
combine_OneHot.head()



train_OneHot = combine_OneHot[combine_OneHot.trainornot == 1]


train_OneHot = train_OneHot.drop(columns=['trainornot'])


test_OneHot = combine_OneHot[combine_OneHot.trainornot == 0]

test_OneHot = test_OneHot.drop(columns=['trainornot'])



train_OneHot.to_csv('D:/Users/p10260458/Desktop/玉山房價預測/新增資料夾/data/train_OneHot.csv', encoding='utf-8', index=False)


test_OneHot.to_csv('D:/Users/p10260458/Desktop/玉山房價預測/新增資料夾/data/test_OneHot.csv', encoding='utf-8', index=False)