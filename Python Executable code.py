#!/usr/bin/env python
# coding: utf-8

# In[56]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from imblearn.over_sampling import SMOTENC


# In[57]:


#Read text file into pandas DataFrame by mentioning the symbol for missing values
missing_values = ["?"]
df = pd.read_csv("./processed.cleveland (2).data",header=None,na_values=missing_values)

#Columns headers
df.columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]

df


# In[58]:


#summary of the data
df.info()


# In[59]:


#Rows and Columns of the data
df.shape


# #initial Data Preprocessing

# In[60]:


#Convert [1,2,3] values of the target variable as 1 while keeping 0 as zero to perform binary classification for the 
#presence of heart diesease

new_target = []

for index,row in df.iterrows():
    if row["target"] > 0:
        new_target.append(1)
    else:
        new_target.append(0)

df["ntarget"] = new_target
df["ntarget"] = df["ntarget"].astype("category")
#Remove the target attribute
df.drop("target",axis=1,inplace = True)
df


# In[61]:


#Convert other categorical variables to numpy categories 
icategoricals = ["sex","cp","fbs","restecg","exang","slope","ca","thal","ntarget"]
categoricals = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]

def convert_to_categorical(categoricals,xcolumn):
    for i in categoricals:
        xcolumn[i] = xcolumn[i].astype("category")
    return xcolumn

df = convert_to_categorical(categoricals=icategoricals,xcolumn=df)
df.info()


# In[62]:


#Finding out missing values of the dataset
df.isna().sum()


# In[63]:


#Heatmap of the missing values of the dataset
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[64]:


#Remove the rows with missing values
df.dropna(inplace=True)
df


# In[65]:


#Check if there are any duplicated values
df.duplicated().sum()


# In[66]:


#Reset Index after the removal
df = df.reset_index(drop=True)
df


# In[67]:


df.shape


# In[68]:


#Statistics of the numerical data attributes
df.describe()


# #Data Visualization 

# In[69]:


#Target Column Distribution
df["ntarget"].value_counts()


# In[70]:


#Count of how many people have heart diesease or not
sns.countplot(x="ntarget",data=df)


# In[71]:


#Function to draw histograms of the numerical data attributes
def histogram(ax,column,xlabel,ylabel = "frequency",n_bins=10):
    ax.hist(df[column], n_bins,ec="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title= xlabel+ " Histogram"
    ax.set_title(title)


# In[72]:


#Histograms of numerical attributes
fig, ((ax0, ax1), (ax2, ax3),(ax4,ax5)) = plt.subplots(nrows=3, ncols=2,figsize=(18,16))

histogram(ax=ax0,column="age",xlabel="Age")
histogram(ax=ax1,column="trestbps",xlabel="Trestbps")
histogram(ax=ax2,column="chol",xlabel="Chol")
histogram(ax=ax3,column="thalach",xlabel="Thalach")
histogram(ax=ax4,column="oldpeak",xlabel="Oldpeak")


plt.tight_layout()

plt.show()


# In[73]:


#Function to draw boxplots of the numerical data attributes against the heart diesease detection
def boxplots(ax,column,xlabel):
    sns.boxplot(x=column,data=df,ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("value")
    title= xlabel+ " Boxplot"
    ax.set_title(title)


# In[74]:


#Boxplots of numerical attributes
fig, ((ax0, ax1), (ax2, ax3),(ax4,ax5)) = plt.subplots(nrows=3, ncols=2,figsize=(18,16))

boxplots(ax=ax0,column="age",xlabel="Age")
boxplots(ax=ax1,column="trestbps",xlabel="Trestbps")
boxplots(ax=ax2,column="chol",xlabel="Chol")
boxplots(ax=ax3,column="thalach",xlabel="Thalach")
boxplots(ax=ax4,column="oldpeak",xlabel="Oldpeak")

plt.tight_layout()

plt.show()


# In[75]:


#Numeric attributes 
numeric = ["age","trestbps","chol","thalach","oldpeak"]
dfin = df[numeric]

#Outlier Removal 
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
dfin = dfin[~((dfin < (Q1 - 1.5 * IQR)) |(dfin > (Q3 + 1.5 * IQR))).any(axis=1)]

dfs = df.copy()
df = dfs[dfs.index.isin(dfin.index.values)]
df = df.reset_index(drop=True)
df


# In[76]:


#Boxplot to confirm outliers were removed
fig, ((ax0, ax1), (ax2, ax3),(ax4,ax5)) = plt.subplots(nrows=3, ncols=2,figsize=(18,16))

boxplots(ax=ax0,column="age",xlabel="Age")
boxplots(ax=ax1,column="trestbps",xlabel="Trestbps")
boxplots(ax=ax2,column="chol",xlabel="Chol")
boxplots(ax=ax3,column="thalach",xlabel="Thalach")
boxplots(ax=ax4,column="oldpeak",xlabel="Oldpeak")

plt.tight_layout()

plt.show()


# In[77]:


#Function to draw countplots of the categorical data attributes
def countplots(ax,column,xlabel):
    sns.countplot(x=column,data=df,ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    title= xlabel+ " Countplot"
    ax.set_title(title)


# In[78]:


#Countplots of categorical columns
fig, ((ax0, ax1), (ax2, ax3),(ax4, ax5),(ax6, ax7)) = plt.subplots(nrows=4, ncols=2,figsize=(18,16))
    
countplots(ax=ax0,column="sex",xlabel="Sex")
countplots(ax=ax1,column="cp",xlabel="CP")
countplots(ax=ax2,column="fbs",xlabel="FBS")
countplots(ax=ax3,column="restecg",xlabel="RestECG")
countplots(ax=ax4,column="exang",xlabel="Exang")
countplots(ax=ax5,column="slope",xlabel="Slope")
countplots(ax=ax6,column="ca",xlabel="Ca")
countplots(ax=ax7,column="thal",xlabel="Thal")


plt.tight_layout()

plt.show()


# In[79]:


#Function to draw countplots of the numerical data attributes against the heart diesease detection
def countplots_with_target(ax,column,xlabel):
    sns.countplot(x=column,data=df,ax=ax,hue="ntarget")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    title= xlabel+ " Countplot"
    ax.set_title(title)


# In[80]:


#Countplots of the numerical data attributes against the heart diesease detection
fig, ((ax0, ax1), (ax2, ax3),(ax4, ax5),(ax6, ax7)) = plt.subplots(nrows=4, ncols=2,figsize=(18,16))

countplots_with_target(ax=ax0,column="sex",xlabel="Sex")
countplots_with_target(ax=ax1,column="cp",xlabel="CP")
countplots_with_target(ax=ax2,column="fbs",xlabel="FBS")
countplots_with_target(ax=ax3,column="restecg",xlabel="RestECG")
countplots_with_target(ax=ax4,column="exang",xlabel="Exang")
countplots_with_target(ax=ax5,column="slope",xlabel="Slope")
countplots_with_target(ax=ax6,column="ca",xlabel="Ca")
countplots_with_target(ax=ax7,column="thal",xlabel="Thal")

plt.tight_layout()

plt.show()


# In[ ]:


#Intial Feature Selection


# In[81]:


#Correlation matrix
correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.heatmap(correlation_matrix,annot=True,linewidths=0.5,cmap='viridis');


# In[82]:


#Function to seperate x and y variables
def seperate_X_and_Y(df):
    xa= df.iloc[:,:-1]
    ya = df["ntarget"]
    return (xa,ya)


# In[83]:


x,y = seperate_X_and_Y(df)
x


# In[349]:


x.info()


# In[350]:


#intial Feature selection using f_class_if
def intial_feature_selection(sf):
    bf = SelectKBest(score_func=sf, k=13)
    fit = bf.fit(x,y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(x.columns)
    feat_scores = pd.concat([df_columns, df_scores],axis=1)
    feat_scores.columns = ['Feature_Name','Score']
    print(feat_scores.nlargest(13,'Score'))


intial_feature_selection(f_classif)


# In[351]:


#intial Feature selection using chi score
intial_feature_selection(chi2)


# In[ ]:


#SVM


# In[87]:


#Function to do label encoding of categorical columns
categoricals = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]

def labelencoding(categoricals,xcolumn):
    for i in categoricals:
        xcolumn[i] = xcolumn[i].cat.codes.astype("category")
    return xcolumn


# In[353]:


x = labelencoding(categoricals=categoricals,xcolumn=x)
x


# In[354]:


#Scaling the data

sc = StandardScaler()
cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
x[cols] = sc.fit_transform(x[cols])
x


# In[355]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109) # 70% training and 30% test


# In[356]:


#Initial SVM model


#SVM Classifier
clf = svm.SVC(kernel='linear')

#Train SVM Classifier
clf.fit(X_train, y_train)

#Predict using SVM Classifier
y_pred = clf.predict(X_test)

# Model Accuracy Score 
print(f"Accuracy \n{metrics.accuracy_score(y_test, y_pred)*100} %\n" )

# Model Confusion Matrix
print(f"Confusion Matrix\n{metrics.confusion_matrix(y_test, y_pred)}\n" )

# Model Classification Report
print("classification report\n\n",metrics.classification_report(y_test, y_pred))


# In[357]:


#Further Feature Selection using l1 penalty
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(X_train, y_train)
model = SelectFromModel(lsvc, prefit=True)


# In[358]:


model.get_support()


# In[359]:


x.shape


# In[360]:


#Finding out the selected features
selected_feat = X_train.columns[(model.get_support())]
selected_feat


# In[361]:


xnew = x[selected_feat]
xnew.info()


# In[362]:


y.value_counts()


# In[364]:


#Oversampling 1 values to make the target class distribution equal with SMOTENC
from imblearn.over_sampling import SMOTE
sm = SMOTENC(random_state=42, categorical_features=[1, 2,5,6,8,10,11,12])
xss, yss = sm.fit_resample(x, y)

yss.value_counts()


# In[365]:


X_sstrain, X_sstest, y_sstrain, y_sstest = train_test_split(xss, yss, test_size=0.3,random_state=109) # 70% training and 30% test


# In[366]:


#Traing the model with after eqaulling the class distribution with SMOTENC

clfsm = svm.SVC(kernel='linear') 
clfsm.fit(X_sstrain, y_sstrain)
y_predsm = clf.predict(X_sstest)

# Model Accuracy Score 
print(f"Accuracy \n{metrics.accuracy_score(y_sstest, y_predsm)*100} %\n" )

# Model Confusion Matrix
print(f"Confusion Matrix\n{metrics.confusion_matrix(y_sstest, y_predsm)}\n" )

# Model Classification Report
print("classification report\n\n",metrics.classification_report(y_sstest, y_predsm))


# In[367]:


#Hyperparameter tuning with 10 fold crossvalidation to find the best parameters

svm_clf = svm.SVC(kernel='linear')

params = {"C":(0.1, 0.5, 1, 2, 5, 10, 20, 40, 60, 100), 
          "gamma":(0.00001,0.0001,0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1), 
          "kernel":('linear', 'poly', 'rbf')}

svm_cv = GridSearchCV(svm_clf, params, n_jobs=-1, cv=10, verbose=1, scoring="accuracy")
svm_cv.fit(x, y)
best_params = svm_cv.best_params_
print(f"Best params: {best_params}")
best_score = svm_cv.best_score_
print(f"Best score: {best_score}")



# In[368]:


#Traing the model with the best parameters
svm_clf = svm.SVC(C=60,gamma=0.0001,kernel="rbf")
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)


# Model Accuracy Score 
print(f"Accuracy \n{metrics.accuracy_score(y_test, y_pred)*100} %\n" )

# Model Confusion Matrix
print(f"Confusion Matrix\n{metrics.confusion_matrix(y_test, y_pred)}\n" )

# Model Classification Report
print("classification report\n\n",metrics.classification_report(y_test, y_pred))


# In[369]:


#Plot Confusion Matrix of SVM model
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Greens')

ax.set_title('Confusion Matrix of Random Forest model\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels([0,1])
ax.yaxis.set_ticklabels([0,1])


plt.show()


# In[253]:


#Random Forest


# In[258]:


#Selecting X and Y for the random forest model
xr,yr = seperate_X_and_Y(df)
xr


# In[259]:


#Label encoding the categorical columns
xr = labelencoding(categoricals=categoricals,xcolumn=xr)
xr


# In[260]:


# Split dataset into training set and test set
X_rtrain, X_rtest, y_rtrain, y_rtest = train_test_split(xr, yr, test_size=0.3,random_state=109) # 70% training and 30% test


# In[302]:


#Intial Random Forest Model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,random_state=10)
model.fit(X_rtrain, y_rtrain)
y_rpred = model.predict(X_rtest)

# Model Accuracy Score 
print(f"Accuracy \n{metrics.accuracy_score(y_rtest, y_rpred)*100} %\n" )

# Model Confusion Matrix
print(f"Confusion Matrix\n{metrics.confusion_matrix(y_rtest, y_rpred)}\n" )

# Model Classification Report
print("classification report\n\n",metrics.classification_report(y_rtest, y_rpred))


# In[129]:


#Further Feature Selection using step forward feature selection Wrapper
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier

# Build RF classifier to use in feature selection
modeln = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Build step forward feature selection
sfs1 = sfs(modeln,
           k_features=13,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)

# Perform SFFS
sfs1 = sfs1.fit(xr, yr)


# In[130]:


#Selected Features
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)


# In[131]:


#Accuracy score using the selected features
print(sfs1.k_score_)


# In[133]:


#Selected Feature Names
print(sfs1.k_feature_names_)


# In[291]:


#Oversampling 1 values to make the target class distribution equal with SMOTENC
from imblearn.over_sampling import SMOTE
sm = SMOTENC(random_state=42, categorical_features=[1, 2,5,6,8,10,11,12])
xss, yss = sm.fit_resample(xr, yr)
# summarize the new class distribution
yss.value_counts()


# In[292]:


X_sstrain, X_sstest, y_sstrain, y_sstest = train_test_split(xss, yss, test_size=0.3,random_state=109) # 70% training and 30% test


# In[303]:


#Traing the model with after eqaulling the class distribution with SMOTENC

model = RandomForestClassifier(n_estimators=100,random_state=10) 
model.fit(X_sstrain, y_sstrain)
y_predsm = model.predict(X_sstest)

# Model Accuracy Score 
print(f"Accuracy \n{metrics.accuracy_score(y_sstest, y_predsm)*100} %\n" )

# Model Confusion Matrix
print(f"Confusion Matrix\n{metrics.confusion_matrix(y_sstest, y_predsm)}\n" )

# Model Classification Report
print("classification report\n\n",metrics.classification_report(y_sstest, y_predsm))


# In[304]:


#Hyper Parameter Tuning of the Random Forest Model with 10 fold crossvalidation

n_estimators = [50,250,500,1000]
max_depth = [2, 3, 5, 10]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

params_grid = {
    'n_estimators': n_estimators, 
    'max_depth': max_depth, 
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
              }

rf_clf = RandomForestClassifier(n_estimators=100,random_state=10)
rf_cv = GridSearchCV(rf_clf, params_grid, scoring="accuracy", cv=10, verbose=1, n_jobs=-1)
rf_cv.fit(xr, yr)
best_params = rf_cv.best_params_
print(f"Best parameters: {best_params}")
best_score = rf_cv.best_score_
print(f"Best Score: {best_score}")


# In[306]:


#Final Random Forest Model with the best parameters
rf_clf = RandomForestClassifier(n_estimators=1000,max_depth=5,min_samples_leaf=1,min_samples_split=5,
                                random_state=10)
rf_clf.fit(X_rtrain, y_rtrain)
y_rpred = rf_clf.predict(X_rtest)

# Model Accuracy Score
print(f"Accuracy \n{metrics.accuracy_score(y_rtest, y_rpred)*100} %\n" )

#Model Confusion Matrix
print(f"Confusion Matrix\n{metrics.confusion_matrix(y_rtest, y_rpred)}\n" )

#Model Classification Report 
print("classification report\n\n",metrics.classification_report(y_rtest, y_rpred))


# In[316]:


#Plot Confusion Matrix of Random Forest model
cf_matrix = metrics.confusion_matrix(y_rtest, y_rpred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Greens')

ax.set_title('Confusion Matrix of Random Forest model\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels([0,1])
ax.yaxis.set_ticklabels([0,1])


plt.show()


# In[369]:


#Deep Neural Netweork


# In[114]:


#Selecting X and Y for the Deep neural network
xa,ya = seperate_X_and_Y(df)
xa


# In[115]:


#Scaling the data

sc = StandardScaler()
cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
xa[cols] = sc.fit_transform(xa[cols])
xa


# In[116]:


#Label encoding the data
xa = labelencoding(categoricals=categoricals,xcolumn=xa)
xa


# In[134]:


# Split dataset into training set and test set
X_atrain, X_atest, y_atrain, y_atest = train_test_split(xa, ya, test_size=0.3,random_state=109) # 70% training and 30% test


# In[135]:


#Initial Deep Neural Network 

model = Sequential()
model.add(Dense(16, input_dim=13,kernel_initializer="normal", activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu',kernel_initializer="normal"))
model.add(Dropout(0.25))
model.add(Dense(4, activation='relu',kernel_initializer="normal"))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

model.fit(X_atrain, y_atrain, epochs=200, batch_size=10, verbose = 1,validation_split=0.2)


# In[136]:


pred = model.predict(X_atest)

y_apred = np.where(pred > 0.5, 1,0)

# Model Accuracy Score
print(f"Accuracy \n{metrics.accuracy_score(y_atest, y_apred)*100} %\n" )

# Model Confusion Matrix
print(f"Confusion Matrix\n{metrics.confusion_matrix(y_atest, y_apred)}\n" )

# Model Classification Report
print("classification report\n\n",metrics.classification_report(y_atest, y_apred))


# In[121]:


#Oversampling 1 values to make the target class distribution equal with SMOTENC
from imblearn.over_sampling import SMOTE
sm = SMOTENC(random_state=42, categorical_features=[1, 2,5,6,8,10,11,12])
xss, yss = sm.fit_resample(xa, ya)

X_sstrain, X_sstest, y_sstrain, y_sstest = train_test_split(xss, yss, test_size=0.3,random_state=109) 

#Traing the model with after eqaulling the class distribution with SMOTENC

model.fit(X_sstrain, y_sstrain, epochs=200, batch_size=10, verbose = 1,validation_split=0.2)


# In[122]:



predss = model.predict(X_sstest)

y_apredss = np.where(predss > 0.5, 1,0)

# Model Accuracy Score
print(f"Accuracy \n{metrics.accuracy_score(y_sstest, y_apredss)*100} %\n" )

# Model Confusion Matrix
print(f"Confusion Matrix\n{metrics.confusion_matrix(y_sstest, y_apredss)}\n" )

# Model Classification Report
print("classification report\n\n",metrics.classification_report(y_sstest, y_apredss))


# In[144]:


#Function for hyperparameter tuning of the neural network
def model_builder(hp):
    hp_units1 = hp.Int('units1', min_value=0, max_value=50, step=1)
    hp_units2 = hp.Int('units2', min_value=0, max_value=30, step=1)
    hp_units3 = hp.Int('units3', min_value=0, max_value=15, step=1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model = Sequential()
    model.add(Dense(units= hp_units1, input_dim=13, activation='relu',kernel_initializer="normal"))
    model.add(Dropout(0.25))
    model.add(Dense(units= hp_units2, activation='relu',kernel_initializer="normal"))
    model.add(Dropout(0.25))
    model.add(Dense(units= hp_units3, activation='relu',kernel_initializer="normal"))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
  
    return model


# In[145]:


#Create the tuner 
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kts11')


# In[146]:


#Early Stopping function
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# In[147]:


tuner.search(xa, ya, epochs=20, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. 
The optimal number of units in the first densely-connected layer is {best_hps.get('units1')}
second densely-connected layer is {best_hps.get('units2')}
third densely-connected layer is {best_hps.get('units3')} 
and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")


# In[148]:


#Final Deep Neural Network 
model = Sequential()
model.add(Dense(11, input_dim=13, activation='relu',kernel_initializer="normal"))
model.add(Dropout(0.25))
model.add(Dense(20, activation='relu',kernel_initializer="normal"))
model.add(Dropout(0.25))
model.add(Dense(13, activation='relu',kernel_initializer="normal"))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

model.fit(X_atrain, y_atrain, epochs=200, batch_size=10, verbose = 1,validation_split=0.2)


# In[149]:


pred = model.predict(X_atest)

y_apred = np.where(pred > 0.5, 1,0)

# Model Accuracy Score  
print(f"Accuracy \n{metrics.accuracy_score(y_atest, y_apred)*100} %\n" )

# Model Confusion Matrix
print(f"Confusion Matrix\n{metrics.confusion_matrix(y_atest, y_apred)}\n" )

# Model Classification Report
print("classification report\n\n",metrics.classification_report(y_atest, y_apred))


# In[150]:


#Plot Confusion Matrix of Deep Neural Network model
cf_matrix = metrics.confusion_matrix(y_atest, y_apred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Greens')

ax.set_title('Confusion Matrix of Random Forest model\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels([0,1])
ax.yaxis.set_ticklabels([0,1])


plt.show()


# In[151]:


#Final Deep Neural Network Architecture
model.summary()


# In[ ]:


#Model Comparison


# In[152]:


#Create table to comapre models
data = {'Model': ["SVM", "RF", "DNN"],
        'Accuracy': [90.4,89.2,89.2],
        'Precision': [91,89,89], 
        'Recall': [90,89,89], 
        'F1 Score': [90,89,89]}

modeldf = pd.DataFrame.from_dict(data)
modeldf


# In[153]:


y_pos = np.arange(len(modeldf["Model"]))

# Create barplot for the accuracies of the 3 models 
plt.bar(y_pos,modeldf["Accuracy"],color=['red', 'green', 'blue'] )

plt.xticks(y_pos, modeldf["Model"])
plt.ylabel("Accuracy")
plt.title("Accuracy scores of 3 models")
plt.figure(figsize=(10000,1000))

plt.show()


# In[ ]:




