# smartSense_Assignment
Given Problem Statement for this activity was:
Train a model to analyze why the best and most experienced employees leave prematurely.
Inorder to achieve this I have included exploratory data analysis, visualization on employee churn dataset and also built a GradientBoosting model and along with this I have compared its prediction with that of an AutoRegressive model

**Objective:** 
1) Perform a Primary Analysis and Visualize the data clearly
2) Perform Clustering Analysis to gather if any meaning full patterns
3) Create a Model to predict the likeliness of an employee leaving the company
4) Understanding the importance of different features that help manager in decision makings

**IMPORT LIBRARIES**
```python
#import modules
import pandas  # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
# % matplotlib inline
```
**READ FILE**
```python
data=pandas.read_csv('HRA.csv')
```
**UNDERSTANDING THE DATA**
This is to gather the data insights
```python
data.head()
print (data.describe())
data.dtypes
```
**VISUALIZATION**

This was done to obtain a broader image about how the data is arranged and was done by creating multiple bar graphs 

Firstly by understanding the proportion of employees who left the company 
```python
left_count=data.groupby('left').count()
plt.bar(left_count.index.values, left_count['satisfaction_level'])
plt.xlabel('Employees Left Company')
plt.ylabel('Number of Employees')
plt.show()
data.left.value_counts()
```
Next we plot to check on an average how many projects are alloted to an employee
```python
num_projects=data.groupby('number_project').count()
plt.bar(num_projects.index.values, num_projects['satisfaction_level'])
plt.xlabel('Number of Projects')
plt.ylabel('Number of Employees')
plt.show()
```
Then we check on how many years does an employee usually work in a company
```python
time_spent=data.groupby('time_spend_company').count()
plt.bar(time_spent.index.values, time_spent['satisfaction_level'])
plt.xlabel('Number of Years Spend in Company')
plt.ylabel('Number of Employees')
plt.show()
```
Later, to get a clearer picture about how multiple features depend on each other, we ploted multiple bars into a single plot
```python
import matplotlib.pyplot as plt
pandas.crosstab(data.sales,data.left).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('department')
plt.ylabel('Frequency of Turnover')
plt.savefig('department_bar_chart')
```
```python
table=pandas.crosstab(data.salary, data.left)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart WRT Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.savefig('salary_bar_chart')
```
Instead of using multiple bar graphs or correlation graphs, we can also use a 3d plot to obtain a clearer picture
```python
from mpl_toolkits.mplot3d import Axes3D
sourceid_data1 = data['number_project']
hod_data1 = data['promotion_last_5years']
count_data1 = data['satisfaction_level']
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_trisurf(sourceid_data1, hod_data1, count_data1, cmap='viridis', edgecolor='none')
ax1.set_xlabel('number of projects')
ax1.set_ylabel('promotion in last 5 years')
ax1.set_zlabel('satisfaction level')
ax1.set_title('3D Plot')
```
**CLUSTERING OF DATA**

Now that we have got a clear idea about how the data is, we can also classify/cluster the datapoints based on the employee satisfaction and last evaluation
```python
from sklearn.cluster import KMeans
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['last_evaluation','satisfaction_level']])
    sse.append(km.inertia_)
plt.xlabel('K Values')
plt.ylabel('SSE values')
plt.plot(k_rng,sse)
```
Here from the graph obtained, using knee-elbow rule we can clearly see that the dataset can be divided into 4 clusters, and we do so by:

```python
left_emp =  data[['satisfaction_level', 'last_evaluation']][data.left == 1]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 4, random_state = 0).fit(left_emp)
left_emp['label'] = kmeans.labels_
# Draw scatter plot
plt.scatter(left_emp['satisfaction_level'], left_emp['last_evaluation'], c=left_emp['label'],cmap='Accent')
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.title('4 Clusters of employees who left')
plt.show()
```
**PREDICTION MODELLING**

Before we start prediction, we need conver the convert all columns into numerical data and to do so we use LabelEncoder the following way:
```python
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['salary']=le.fit_transform(data['salary'])
data['sales']=le.fit_transform(data['sales'])
```
Now that we have converted department and all other non numeric columns, we can start building our Model using Gradient Boosting Classifier
```python
X=data[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'sales', 'salary']]
y=data['left']
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Here we have split our data in 80-20% for training and testing, now we are going to predict and further calculate the accuracy, precision and recall
```python
#Import Gradient Boosting Classifier model
from sklearn.ensemble import GradientBoostingClassifier

#Create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

#Train the model using the training sets
gb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gb.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))
```
Now that we obtained an accuracy of 0.98 we are going to predict the same using other methods for checking which would be better

Using AutoRegression method
```python
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
dftest = adfuller(data['left'], autolag='AIC')
print("P value:", dftest[1])
train = X[:len(X)-4]
test = X[len(X)-4:]

# Split the data into train and test sets
train = data['left'][:len(X)-450] 
test = data['left'][-450:]  

# Convert the train and test data to a numeric type
train = pandas.to_numeric(train, errors='coerce')
test = pandas.to_numeric(test, errors='coerce')


test = test.dropna()
leng_train = len(train)
# Specify the number of lags (adjust this value as needed)
lag =1
# Ensure that lags is less than nobs_train
if lag >= leng_train:
    raise ValueError("Number of lags should be less than the number of observations.")
model = AutoReg(train, lags = lag).fit()
predictions = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
```
We are able to get a decent prediction even through AutoRegression with an RMSE value of approx 0.14

**Calculating the Importance of different features**

Inorder to Calculate the importance we use Random Forest Model,
```python
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rf = RandomForestClassifier(
    class_weight="balanced")

# Fit the RF Model
rf = rf.fit(X_train, y_train)



# Define the 10-Fold Cross Validation
kfold = model_selection.KFold(n_splits=10, random_state=None)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
scoring = 'roc_auc'
rf_results = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)
rf_auc = rf_results.mean()
print("The Random Forest AUC: %.3f and the STD is (%.3f)" % (rf_auc, rf_results.std()))
```
Here we are able to notice the predicted values standard deviation to be just 0.003
Now we try to obtain the importance of each feature individually
```python
feature_importances = pandas.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()
feature_importances
```
