import pandas as pd
pd.set_option("max_columns", 60)
pd.set_option("max_rows", 500)
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from scipy.stats import binned_statistic

from sklearn.externals.six import StringIO
from IPython.display import Image, display
import pydotplus

os.chdir(r"C:\Users\Aaron\Google Drive\School Stuff\Spring 2020\Data Science Programming II\project-deliverable-2-vgs-dsp\data")


#TODO: 

data = pd.read_csv("dt_data.csv")
data = data.drop(columns = 'Unnamed: 0')


#Loop to replace all null scores with the average of the existing row scores
for row in range(len(data)):
    #average of score columns rounded to int
    avg = np.around(np.nanmean(data.iloc[row, range(15,21)]))
    #loops over score columns
    for col in range(15, 21):
        if np.isnan(data.iloc[row, col]):
            data.iloc[row, col] = avg


#Replacing missing ESRB ratings with Most occuring rating
imp_mode = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
data_imp = pd.DataFrame(imp_mode.fit_transform(data))
data_imp.columns = data.columns
data_imp.index = data.index
data_imp.sales = data_imp.sales.astype('int64')

#Print amounts of null values per column
data_imp.isna().sum()

#Binning units_total


data_imp['sales'].max()
data_imp['sales'].min()
#data['sales'].plot.hist(alpha=0.5)
#plt.show()


######################################################################
#normal_bin_interval = [0, 17500, 20000, 25000, 32500, 50000, 100000, 250000, 
#                1000000, 2000000, 2750000, 3750000, 5000000, 7500000, 
#                10000000, 20000000]
#
#
#
#bin_counts, bin_edges, binnum = binned_statistic(data_imp['sales'],
#                                                data_imp['sales'],
#                                                 statistic='count',
#                                                 bins=normal_bin_interval)

#bin_counts
#
#bin_edges
#
#bin_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#
#sales_category = pd.cut(data_imp['sales'], normal_bin_interval, right=False, retbins=False, labels=bin_labels)
#sales_category.name = 'sales_categ'
#
#data_imp = pd.concat([data_imp, sales_category], axis=1)
#
#data_imp['sales_categ'].value_counts().sort_index().plot(kind='bar')
#plt.show()


#######################################################
width_bin_interval = [0, 60500, 215000, 1150000, 
                      20000000]

bin_counts, bin_edges, binnum = binned_statistic(data_imp['sales'], 
                                                 data_imp['sales'],
                                                 statistic='count',
                                                 bins=width_bin_interval)

bin_counts

bin_edges

bin_labels = [0, 1, 2, 3]

sales_category = pd.cut(data_imp['sales'], width_bin_interval, right=False, retbins=False, labels=bin_labels)
sales_category.name = 'sales_categ'

data_imp = pd.concat([data_imp, sales_category], axis=1)

data_imp['sales_categ'].value_counts().sort_index().plot(kind='bar')
plt.show()

os.chdir(r'C:\Users\Aaron\Google Drive\School Stuff\Spring 2020\Data Science Programming II\project-deliverable-2-vgs-dsp\visualizations')

data_imp = data_imp.drop(['title'], axis = 1)
#################################################
#
#
#       Classification Trees
#
#
#################################################
#DT with all variables
#Accuracy: approx. 37.8%

predictors = data_imp.drop(['sales', 'sales_categ'], axis = 1)

target = data_imp['sales_categ']

scores = []

for i in range(1, 250):
    train, test, y_train, y_test = train_test_split(predictors, target, test_size=0.1)
    vg_tree = tree.DecisionTreeClassifier(max_leaf_nodes = 4, min_samples_split=20, min_samples_leaf=1,
                                      max_depth = 4).fit(train, y_train)
    predicted = vg_tree.predict(test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    scores.append(accuracy)
    i+=1

full_score = np.around(np.mean(scores)*100, decimals=4)
print("Accuracy of Full Tree: ", full_score, "%", sep='')

dot_data = tree.export_graphviz(vg_tree, feature_names=predictors.columns.values, 
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('dt_graph.png')


print(metrics.classification_report(y_test, predicted))
        
cm = metrics.confusion_matrix(y_test, predicted)
print(cm)
plt.matshow(cm)
plt.title('Confusion Matrix: Full Decision Tree')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.savefig("dt_cm.png")
############################################
#DT with consumer info
#Accuracy: approx 35.9%

ci_data = data_imp.drop(['metascore', 'user_score_x10', 'igdb_member_rating', 'igdb_critic_rating',
                               'ign_x10', 'mgAverage'], axis = 1)
ci_predictors = ci_data.drop(['sales', 'sales_categ'], axis = 1)

ci_target = ci_data.sales_categ

ci_scores = []

for i in range(1,1000):
    ci_train, ci_test, ci_y_train, ci_y_test = train_test_split(ci_predictors, ci_target, test_size=0.1)
    ci_tree = tree.DecisionTreeClassifier(max_leaf_nodes = 9, min_samples_split=10, min_samples_leaf=10,
                                      max_depth = 10).fit(ci_train, ci_y_train)
    ci_predicted = ci_tree.predict(ci_test)
    accuracy = metrics.accuracy_score(ci_y_test, ci_predicted)
    ci_scores.append(accuracy)
    i += 1

ci_score = np.around(np.mean(ci_scores)*100, decimals=4)
print("Average of Consumer Info Tree: ", ci_score, "%", sep='')

dot_data = tree.export_graphviz(ci_tree, feature_names=ci_predictors.columns.values,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('dt_ci_graph.png')

print(metrics.classification_report(ci_y_test, ci_predicted))
ci_cm = metrics.confusion_matrix(ci_y_test, ci_predicted)
print(ci_cm)
plt.matshow(ci_cm)
plt.title('Confusion Matrix: Consumer Info Decision Tree')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.savefig("dt_ci_cm.png")
############################################
#DT with review info
#Accuracy: 37.5%

rev_data = data_imp.drop(['console_PS4', 'console_XBox1',
                                'console_Switch', 'genre_Fighting', 'genre_Misc',
                                'genre_Platform', 'genre_Racing', 'genre_RPG',
                                'genre_Shooter', 'genre_Sports', 
                                'genre_Action_Adventure', 'player_vector',
                                'esrb_vector'], axis = 1)

rev_predictors = rev_data.drop(['sales', 'sales_categ'], axis=1)

rev_target = rev_data.sales_categ

rev_scores = []

for i in range(1, 1000):
    rev_train, rev_test, rev_y_train, rev_y_test = train_test_split(rev_predictors, rev_target, test_size=0.10)
    rev_tree = tree.DecisionTreeClassifier(max_leaf_nodes = 10, min_samples_split = 10, min_samples_leaf = 10,
                                           max_depth = 8).fit(rev_train, rev_y_train)
    rev_predicted = rev_tree.predict(rev_test)
    accuracy = metrics.accuracy_score(rev_y_test, rev_predicted)
    rev_scores.append(accuracy)
    i += 1

rev_score = np.around(np.mean(rev_scores)*100, decimals=4)
print("Average of Review Info Tree: ", rev_score, "%", sep='')

dot_data = tree.export_graphviz(rev_tree, feature_names=rev_predictors.columns.values,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('dt_rev_graph.png')

print(metrics.classification_report(rev_y_test, rev_predicted))

rev_cm = metrics.confusion_matrix(rev_y_test, rev_predicted)
print(rev_cm)
plt.matshow(rev_cm)
plt.title('Confusion Matrix: Reviews Decision Tree')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.savefig("dt_rev_cm.png")
##################################################
#
#
#    Regression Trees
#
#
##################################################
#Full Regression DT
#reg_target = data_imp.sales

#reg_train, reg_test, reg_y_train, reg_y_test = train_test_split(predictors, reg_target, test_size=0.1)

#full_reg_tree = tree.DecisionTreeRegressor().fit(reg_train, reg_y_train)

#reg_predicted = full_reg_tree.predict(reg_test)

#reg_r2_score = metrics.r2_score(reg_y_test, reg_predicted)
#reg_mae = metrics.mean_absolute_error(reg_y_test, reg_predicted)

#dot_data = tree.export_graphviz(full_reg_tree, feature_names=predictors.columns.values,
#                                filled=True, rounded=True,
#                                special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_png('reg_dt_full.png')

#########################################################
#Plotting Tree Scores

plt.clf()

sns.set(style='whitegrid')
scores = pd.DataFrame(data = {'full_tree': [full_score], 'customer_info_score': [ci_score], 'review_info_score': [rev_score]})
ax = sns.barplot(data=scores)
ax.set(xlabel='Tree', ylabel='Classification Success Rate %')
ax.set(ylim=(20,40))
#ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
fig = ax.get_figure()
plt.figure(figsize=(8,4))
fig.savefig('tree_comparison.png')