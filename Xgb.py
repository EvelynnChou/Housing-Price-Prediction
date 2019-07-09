#!/remote/vgverdivia1/powu/anaconda3/bin/python3.7
#https://www.datacamp.com/community/tutorials/xgboost-in-python
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
#output file
import pickle
from sklearn.externals import joblib 
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def read_csv(csv_name):
  features = pd.read_csv(csv_name)
  return features


def print_csv_info(features):
  print('---------------------------------------------------')
  print('<First 5 rows>')
  print(features.head(5))
  
  print('---------------------------------------------------')
  print('<csv shape>')
  print(features.shape)
  
  print('---------------------------------------------------')
  print('<column distribution>')
  # Descriptive statistics for each column
  print(features.describe())


def split_data(features, labels, feature_list):
# Labels are the values we want to predict
  labels = np.array(features['total_price'])
# Remove the labels from the features
# axis 1 refers to the columns
  features = features.drop('total_price', axis = 1)
# Saving feature names for later use
  feature_list = list(features.columns)
# Convert to numpy array
  features = np.array(features)
  return features

def split_into_test_and_train(features, labels):
# Split the data into training and testing sets
  train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

  print('Training Features Shape:', train_features.shape)
  print('Training Labels Shape:', train_labels.shape)
  print('Testing Features Shape:', test_features.shape)
  print('Testing Labels Shape:', test_labels.shape)
   

def train_data(train_features, train_labels, test_features, test_labels):
# Instantiate model with 1000 decision trees
  print('---------------------------------------------------')
  print('<Start to train data, the n_estimators is:>')
  estimators = 2000
  print(estimators)
#rf = RandomForestRegressor(n_estimators = estimators, random_state = 42)
# Train the model on training data
#rf.fit(train_features, train_labels);
  xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,\
                            max_depth = 5, alpha = 200, n_estimators = estimators)
  xg_reg.fit(train_features, train_labels)
  predictions = xg_reg.predict(test_features)

# Use the forest's predict method on the test data
#  predictions = rf.predict(test_features)



# Calculate the absolute errors
  errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
  print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
  mape = 100 * (errors / test_labels)
# Calculate and display accuracy
  accuracy = 100 - np.mean(mape)
  print('Accuracy:', round(accuracy, 2), '%.')

  rmse = np.sqrt(mean_squared_error(test_labels, predictions))
  print("RMSE: %f" % (rmse))

  return xg_reg


def get_importances(rf, feature_list):
# Get numerical feature importances
  importances = list(rf.feature_importances_)
# List of tuples with variable and importance
  feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
  feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
  [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# select the top 100 data and dump to log file
  feature_importances = feature_importances[0:99]
  with open('importance.log', 'w') as f:
    out_str = "features = features[["
    for item in feature_importances:
      out_str += "'" + item[0] + "',"
    out_str = out_str[0:-1]
    out_str += ",'total_price']]"
    f.write("%s" % out_str)
  

def rerun_with_new_data(features):
# New random forest with only the two most important variables
  rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
  important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
  train_important = train_features[:, important_indices]
  test_important = test_features[:, important_indices]
# Train the random forest
  rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
  predictions = rf_most_important.predict(test_important)
  errors = abs(predictions - test_labels)
# Display the performance metrics
  print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
  mape = np.mean(100 * (errors / test_labels))
  accuracy = 100 - mape
  print('Accuracy:', round(accuracy, 2), '%.')





features = read_csv('new_train_no_chinese.csv')

#print_csv_info(features)

features = features.drop('building_id', axis = 1)
#features = features[['building_area','III_10000','parking_price','land_area','V_10000',\
#                    'VI_10000','XIII_10000','village_income_median','divorce_rate','I_10000',\
#                    'II_10000','VII_10000','XI_5000','XII_10000','XIII_5000',\
#                    'doc_rate','village','N_50','III_5000','VI_MIN',\
#                    'VIII_10000','XI_10000','XII_250',\
#                    'total_price']]
#features = features[[\
#           'building_area',\
#           'parking_price',\
#           'land_area',\
#           'XIII_10000',\
#           'II_10000',\
#           'III_10000',\
#           'V_10000',\
#           'VII_10000',\
#           'XIII_5000',\
#           'I_10000',\
#           'IX_10000',\
#           'XI_10000',\
#           'village_income_median',\
#           'doc_rate',\
#           'II_5000',\
#           'IV_10000',\
#           'VI_10000',\
#           'VII_500',\
#           'VIII_10000',\
#           'IX_250',\
#           'X_500',\
#           'X_10000',\
#           'XII_10000',\
#           'XII_MIN',\
#           'XIV_MIN',\
#           'total_price'\
#            ]]
features = features[['building_area','VII_10000','village_income_median','III_10000','parking_price','village','XII_250','XIII_10000','land_area','lon','VII_100','IX_10000','XI_100','town_population','doc_rate','II_1000','IV_1000','V_index_50','XII_50','XIII_1000','city','town_population_density','building_material','txn_dt','total_floor','building_type','building_use','building_complete_dt','parking_way','parking_area','txn_floor','lat','town_area','master_rate','bachelor_rate','jobschool_rate','highschool_rate','junior_rate','elementary_rate','born_rate','death_rate','marriage_rate','divorce_rate','N_50','N_500','N_1000','N_5000','N_10000','I_10','total_price']]
features = features.fillna(0)

print_csv_info(features)

# Labels are the values we want to predict
labels = np.array(features['total_price'])
# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('total_price', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


#rf = train_data(train_features, train_labels, test_features, test_labels)
#get_importances(rf, feature_list)

xgb = train_data(train_features, train_labels, test_features, test_labels)

# Save the model as a pickle in a file 
#joblib.dump(rf, 'jet.pkl') 
