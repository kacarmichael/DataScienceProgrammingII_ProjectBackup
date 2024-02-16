import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn import metrics

#Neural Network
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from scipy.stats import binned_statistic

#Multiple Regression
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

#dir = r"C:\Users\arkad\OneDrive\Desktop\Assignments\Spring'20\MSIS 5223 Programming For Data Science and Analytics II\Project\project-deliverable-2-vgs-dsp\data"
#os.chdir(dir)

os.chdir(r"C:\Users\Aaron\Google Drive\School Stuff\Spring 2020\Data Science Programming II\project-deliverable-2-vgs-dsp\data")

vg_data = pd.read_csv('dt_data.csv')
vg_data.columns
vg_data.dtypes
vg_data = vg_data.drop(columns = 'Unnamed: 0')

#Count null values
vg_data.isna().sum()

#Replace missing score values with the average of all scores per row
for row in range(len(vg_data)):
    #average of score columns rounded to int
    avg = np.around(np.nanmean(vg_data.iloc[row, range(15,21)]))
    #loops over score columns
    for col in range(15, 21):
        if np.isnan(vg_data.iloc[row, col]):
            vg_data.iloc[row, col] = avg


#Replacing missing ESRB ratings with Most occuring rating
imp_mode = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
vg_data_imp = pd.DataFrame(imp_mode.fit_transform(vg_data))
vg_data_imp.columns = vg_data.columns
vg_data_imp.index = vg_data.index
vg_data_imp.sales = vg_data_imp.sales.astype('int64')

#Print amounts of null values per column
vg_data_imp.isna().sum()

#Binning sales
width_bin_interval = [0, 60500, 215000, 1150000, 
                      20000000]

bin_counts, bin_edges, binnum = binned_statistic(vg_data_imp['sales'], 
                                                 vg_data_imp['sales'],
                                                 statistic='count',
                                                 bins=width_bin_interval)

bin_counts

bin_edges

bin_labels = [0, 1, 2, 3]

sales_category = pd.cut(vg_data_imp['sales'], width_bin_interval, right=False, retbins=False, labels=bin_labels)
sales_category.name = 'sales_categ'

vg_data_imp = pd.concat([vg_data_imp, sales_category], axis=1)

vg_data_imp['sales_categ'].value_counts().sort_index().plot(kind='bar')
plt.show()

os.chdir(r'C:\Users\Aaron\Google Drive\School Stuff\Spring 2020\Data Science Programming II\project-deliverable-2-vgs-dsp\visualizations')

vg_data_imp = vg_data_imp.drop(['title'], axis = 1)

###########################################
predictors = vg_data_imp.drop(['sales', 'sales_categ'], axis = 1)
target = vg_data_imp.sales_categ


nn1_scores = []
nn2_scores = []
nn3_scores = []
nn4_scores = []
scaler = preprocessing.StandardScaler()

for i in range(1, 10):
    vg_data_train, vg_data_test, y_train, y_test = train_test_split(predictors, 
                                                                      target, 
                                                                      test_size=0.1)
    scaler.fit(vg_data_train)
    vg_data_train_std = scaler.transform(vg_data_train)
    vg_data_test_std = scaler.transform(vg_data_test)
    nnclass1 = MLPClassifier(activation='logistic', solver='sgd', 
                         hidden_layer_sizes=(100,100), max_iter=500)
    nnclass1.fit(vg_data_train_std, y_train)
    nnclass1_pred = nnclass1.predict(vg_data_test_std)
    #cm = metrics.confusion_matrix(y_test, nnclass1_pred)
    #print(cm)
    #plt.matshow(cm)
    #plt.title('Confusion Matrix')
    #plt.xlabel('Actual Value')
    #plt.ylabel('Predicted Value')
    #plt.xticks([0,1,2,3], ['I','II','III','IV'])
    #plt.show()
    nn1_scores.append(metrics.accuracy_score(y_test, nnclass1_pred))
    ###################################################
    nnclass2 = MLPClassifier(activation='relu', solver='sgd',
                         hidden_layer_sizes=(100,100), max_iter=500)
    nnclass2.fit(vg_data_train_std, y_train)
    nnclass2_pred = nnclass2.predict(vg_data_test_std)
    #cm2 = metrics.confusion_matrix(y_test, nnclass2_pred)
    #print(cm2)
    #plt.matshow(cm2)
    #plt.title('Confusion Matrix 2')
    #plt.xlabel('Actual Value')
    #plt.ylabel('Predicted Value')
    #plt.xticks([0,1,2,3], ['I','II','III','IV'])
    #plt.show()
    nn2_scores.append(metrics.accuracy_score(y_test, nnclass2_pred))
    ####################################################
    nnclass3 = MLPClassifier(activation='relu', solver='lbfgs',
                         hidden_layer_sizes=(100,100), max_iter=500)
    nnclass3.fit(vg_data_train_std, y_train)
    nnclass3_pred = nnclass3.predict(vg_data_test_std)
    #cm3 = metrics.confusion_matrix(y_test, nnclass3_pred)
    #print(cm3)
    #plt.matshow(cm3)
    #plt.title('Confusion Matrix 3')
    #plt.xlabel('Actual Value')
    #plt.ylabel('Predicted Value')
    #plt.xticks([0,1,2,3], ['I','II','III','IV'])
    #plt.show()
    nn3_scores.append(metrics.accuracy_score(y_test, nnclass3_pred))
    ###################################################
    nnclass4 = MLPClassifier(activation='logistic', solver='lbfgs',
                         hidden_layer_sizes=(100,100), max_iter=500)
    nnclass4.fit(vg_data_train_std, y_train)
    nnclass4_pred = nnclass4.predict(vg_data_test_std)
    #cm4 = metrics.confusion_matrix(y_test, nnclass2_pred)
    #print(cm4)
    #plt.matshow(cm4)
    #plt.title('Confusion Matrix 4')
    #plt.xlabel('Actual Value')
    #plt.ylabel('Predicted Value')
    #plt.xticks([0,1,2,3], ['I','II','III','IV'])
    #plt.show()
    nn4_scores.append(metrics.accuracy_score(y_test, nnclass4_pred))
    ##################################################


#Accuracy scores of different settings
plt.clf()

nn1_avg = np.around(np.mean(nn1_scores)*100, decimals=4)
nn2_avg = np.around(np.mean(nn2_scores)*100, decimals=4)
nn3_avg = np.around(np.mean(nn3_scores)*100, decimals=4)
nn4_avg = np.around(np.mean(nn4_scores)*100, decimals=4)

nn_setting_scores = pd.DataFrame(data = {'logistic_sgd': [nn1_avg], 'relu_sgd': [nn2_avg], 'relu_lbfgs': [nn3_avg], 'logistic_lbfgs': [nn4_avg]})

sns.set(style='whitegrid')
ax = sns.barplot(data=nn_setting_scores)
ax.set(xlabel='Neural Network', ylabel='Classification Success Rate %')
ax.set(ylim=(10,50))
fig = ax.get_figure()
fig.savefig('nn_setting_comparison.png')

nn1_avg
nn2_avg
nn3_avg
nn4_avg

###################################################################
#
#
#     Testing Different Variable Combinations
#
#
###################################################################
# NN using only customer info

ci_predictors = vg_data_imp.drop(['sales', 'sales_categ', 'metascore', 'user_score_x10', 'igdb_member_rating', 'igdb_critic_rating', 'ign_x10', 'mgAverage'], axis=1)

ci_scores = []

for i in range(1, 10):
    ci_train, ci_test, y_train, y_test = train_test_split(ci_predictors, target, test_size=0.1)
    scaler.fit(ci_train)
    ci_train_std = scaler.transform(ci_train)
    ci_test_std = scaler.transform(ci_test)
    nn_ci = MLPClassifier(activation = 'relu', solver = 'sgd',
                          hidden_layer_sizes = (100,100), max_iter=500)
    nn_ci.fit(ci_train_std, y_train)
    nn_ci_pred = nn_ci.predict(ci_test_std)
    ci_scores.append(metrics.accuracy_score(y_test, nn_ci_pred))

ci_avg = np.around(np.mean(ci_scores)*100, decimals=4)

####################################################################
# NN using review info

rev_predictors = vg_data_imp.drop(['sales', 'sales_categ', 'console_PS4', 'console_XBox1',
                                'console_Switch', 'genre_Fighting', 'genre_Misc',
                                'genre_Platform', 'genre_Racing', 'genre_RPG',
                                'genre_Shooter', 'genre_Sports', 
                                'genre_Action_Adventure', 'player_vector',
                                'esrb_vector'], axis = 1)

rev_scores = []

for i in range(1, 10):
        rev_train, rev_test, y_train, y_test = train_test_split(rev_predictors, target, test_size=0.1)
        scaler.fit(rev_train)
        rev_train_std = scaler.transform(rev_train)
        rev_test_std = scaler.transform(rev_test)
        nn_rev = MLPClassifier(activation = 'relu', solver = 'sgd',
                               hidden_layer_sizes = (100, 100), max_iter=500)
        nn_rev.fit(rev_train_std, y_train)
        nn_rev_pred = nn_rev.predict(rev_test_std)
        rev_scores.append(metrics.accuracy_score(y_test, nn_rev_pred))

rev_avg = np.around(np.mean(rev_scores)*100, decimals=4)
#####################################################################
# Comparing models

final_scores = pd.DataFrame(data = {'Full': [nn2_avg], 'Customer Info': [ci_avg], 'Review Info': [rev_avg]})

plt.clf()

sns.set(style='whitegrid')
ax = sns.barplot(data=final_scores)
ax.set(xlabel='Neural Network', ylabel='Classification Success Rate %')
ax.set(ylim=(10,50))
fig = ax.get_figure()
fig.savefig('nn_predictor_comparison.png')
