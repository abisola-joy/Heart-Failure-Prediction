#!/usr/bin/env python
# coding: utf-8

# In[34]:


#import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import style
from scipy.stats import skew, kurtosis, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, make_scorer, balanced_accuracy_score, mean_squared_error

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_selection import chi2
from sklearn.datasets import make_regression


# In[35]:


#Using numpy to load the file inorder to start the Exploratory Data Analysis
#using pandas to load the file for ML classification
try:
    #loading the file 
    health_data = np.loadtxt('heart_failure_clinical_records_dataset.csv', delimiter=",", skiprows=1)
    health_data_df = pd.read_csv(r'heart_failure_clinical_records_dataset.csv')

    #converting loaded string values to float
    health_data = health_data.astype('float')#there are some decimal values that we lose if the values are converted to int64
    
    #exception to handle file error  
except IOError as ioerr:
    print('File error: ' + str(ioerr))  


# In[36]:


#Defining a function to explore the dataset
def dataset():
    """ function for dataset description """
    
    #empty list to store values
    survived, dead = [],[]
    male, female = [],[]
    
    #getting the values of sex column
    sex = health_data[:,9:10]
    
    #iterating through sex values
    for i in sex:
        if i == 1.0:
            #appending male (1) to male list
            male.append(i)
        elif i == 0.0:
            #appending female (0) to female list
            female.append(i)

    #getting the values of DEATH_EVENT column
    DEATH_EVENT = health_data[:,-1]
    
    #iterating through DEATH_EVENT values
    for i in DEATH_EVENT:
        if i == 1.0:
            #appending dead (1) to dead list
            dead.append(i)
        elif i == 0.0:
            #appending survived (0) to survived list
            survived.append(i)
    
    #printing the length of survived, dead, female and male list to get the total number
    print("\tSUMMARY OF SOME VARIABLES IN THE DATASET")
    print("DEATH_EVENT:\nThe total survived patients are:",(len(survived)))
    print("The total dead patients are:",(len(dead)))
    print()
    print("SEX:\nThe total female patients are:",(len(female)))
    print("The total male patients are:",(len(male)))
    print()
    
    #printing the minimum and maximum age as integer
    print("AGE:\nMinimum:",np.min(health_data[:,0:1].astype('int64')))
    print("Maximum:",np.max(health_data[:,0:1].astype('int64')))
    print()
        
    #printing the minimum and maximum creatinine phosphokinase as integer
    print("CREATININE PHOSPHOKINASE:\nMinimum:",np.min(health_data[:,2:3].astype('int64')))
    print("Maximum:",np.max(health_data[:,2:3].astype('int64')))
    print()
        
    #printing the minimum and maximum ejection fraction as integer
    print("EJECTION FRACTION:\nMinimum:",np.min(health_data[:,4:5].astype('int64')))
    print("Maximum:",np.max(health_data[:,4:5].astype('int64')))
    print()
        
    #printing the minimum and maximum platelets as integer
    print("PLATELETS:\nMinimum:",np.min(health_data[:,6:7].astype('int64')))
    print("Maximum:",np.max(health_data[:,6:7].astype('int64')))
    print()
        
    #printing the minimum and maximum serum creatinine
    print("SERUM CREATININE:\nMinimum:",np.min(health_data[:,7:8]))
    print("Maximum:",np.max(health_data[:,7:8]))
    print()
            
    #printing the minimum and maximum serum sodium as integer
    print("SERUM SODIUM:\nMinimum:",np.min(health_data[:,8:9].astype('int64')))
    print("Maximum:",np.max(health_data[:,8:9].astype('int64')))
    print()
    
    #printing the minimum and maximum time as integer
    print("TIME:\nMinimum time:",np.min(health_data[:,11:12].astype('int64')))
    print("Maximum time:",np.max(health_data[:,11:12].astype('int64')))    


# In[50]:


#Defining a function to check for missing values  
def checking_missing_values():
    """ function to check for missing values """
    #checking if there are missing values in the file using isnull
    print(health_data_df.isnull().sum())
    
    #checking if there are missing values in the file using isnan
    print(np.isnan(health_data))

#Defining a function to return the measures of location
def measures_of_location(choice):
    """ function to return measures of location: mean and median and measures of shape: Skewness and Kurtosis """
    
    #Measures of location: Mean
    if choice.lower() == 'mean':
        #return np.mean(health_data,axis = 0)#This can be printed to make the code more concise       
        #choosing indexing to make the output clearer
        print("age \t\t\t",np.mean(health_data[:,0:1])) #Mean of age
        print("anaemia \t\t",np.mean(health_data[:,1:2])) #Mean of anaemia
        print("creatinine_phosphokinase",np.mean(health_data[:,2:3])) #Mean of creatinine_phosphokinase
        print("diabetes \t\t",np.mean(health_data[:,3:4])) #Mean of diabetes
        print("ejection_fraction \t",np.mean(health_data[:,4:5])) #Mean of ejection_fraction
        print("high_blood_pressure \t",np.mean(health_data[:,5:6])) #Mean of high_blood_pressure
        print("platelets \t\t",np.mean(health_data[:,6:7])) #Mean of platelets
        print("serum_creatinine \t",np.mean(health_data[:,7:8])) #Mean of serum_creatinine
        print("serum_sodium \t\t",np.mean(health_data[:,8:9])) #Mean of serum_sodium
        print("sex \t\t\t",np.mean(health_data[:,9:10])) #Mean of sex
        print("smoking \t\t",np.mean(health_data[:,10:11])) #Mean of smoking
        print("time \t\t\t",np.mean(health_data[:,11:12])) #Mean of time
        print("DEATH_EVENT \t\t",np.mean(health_data[:,-1])) #Mean of DEATH_EVENT
        
    #Measures of location: Median
    elif choice.lower() == 'median':
        #return np.median(health_data,axis = 0)#This can be printed to make the code more concise
        print("age \t\t\t",np.median(health_data[:,0:1])) #Median of age
        print("anaemia \t\t",np.median(health_data[:,1:2])) #Median of anaemia
        print("creatinine_phosphokinase",np.median(health_data[:,2:3])) #Median of creatinine_phosphokinase
        print("diabetes \t\t",np.median(health_data[:,3:4])) #Median of diabetes
        print("ejection_fraction \t",np.median(health_data[:,4:5])) #Median of ejection_fraction
        print("high_blood_pressure \t",np.median(health_data[:,5:6])) #Median of high_blood_pressure
        print("platelets \t\t",np.median(health_data[:,6:7])) #Median of platelets
        print("serum_creatinine \t",np.median(health_data[:,7:8])) #Median of serum_creatinine
        print("serum_sodium \t\t",np.median(health_data[:,8:9])) #Median of serum_sodium
        print("sex \t\t\t",np.median(health_data[:,9:10])) #Median of sex
        print("smoking \t\t",np.median(health_data[:,10:11])) #Median of smoking
        print("time \t\t\t",np.median(health_data[:,11:12])) #Median of time
        print("DEATH_EVENT \t\t",np.median(health_data[:,-1])) #Median of DEATH_EVENT
        
    #Measures of Shape: Skewness
    elif choice.lower() == 'skew':
        print("age \t\t\t",skew(health_data[:,0:1], bias=False)) #skewness of age
        print("anaemia \t\t",skew(health_data[:,1:2], bias=False)) #skewness of anaemia
        print("creatinine_phosphokinase",skew(health_data[:,2:3], bias=False)) #skewness of creatinine_phosphokinase
        print("diabetes \t\t",skew(health_data[:,3:4], bias=False)) #skewness of diabetes
        print("ejection_fraction \t",skew(health_data[:,4:5], bias=False)) #skewness of ejection_fraction
        print("high_blood_pressure \t",skew(health_data[:,5:6], bias=False)) #skewness of high_blood_pressure
        print("platelets \t\t",skew(health_data[:,6:7], bias=False)) #skewness of platelets
        print("serum_creatinine \t",skew(health_data[:,7:8], bias=False)) #skewness of serum_creatinine
        print("serum_sodium \t\t",skew(health_data[:,8:9], bias=False)) #skewness of serum_sodium
        print("sex \t\t\t",skew(health_data[:,9:10], bias=False)) #skewness of sex
        print("smoking \t\t",skew(health_data[:,10:11], bias=False)) #skewness of smoking
        print("time \t\t\t",skew(health_data[:,11:12], bias=False)) #skewness of time
        print("DEATH_EVENT \t\t",skew(health_data[:,-1], bias=False)) #skewness of DEATH_EVENT
        
    #Measures of Shape: Kurtosis
    elif choice.lower() == 'kurt':    
        print("age \t\t\t",kurtosis(health_data[:,0:1], bias=False)) #kurtosis of age
        print("anaemia \t\t",kurtosis(health_data[:,1:2], bias=False)) #kurtosis of anaemia
        print("creatinine_phosphokinase",kurtosis(health_data[:,2:3], bias=False)) #kurtosis of creatinine_phosphokinase
        print("diabetes \t\t",kurtosis(health_data[:,3:4], bias=False)) #kurtosis of diabetes
        print("ejection_fraction \t",kurtosis(health_data[:,4:5], bias=False)) #kurtosis of ejection_fraction
        print("high_blood_pressure \t",kurtosis(health_data[:,5:6], bias=False)) #kurtosis of high_blood_pressure
        print("platelets \t\t",kurtosis(health_data[:,6:7], bias=False)) #kurtosis of platelets
        print("serum_creatinine \t",kurtosis(health_data[:,7:8], bias=False)) #kurtosis of serum_creatinine
        print("serum_sodium \t\t",kurtosis(health_data[:,8:9], bias=False)) #kurtosis of serum_sodium
        print("sex \t\t\t",kurtosis(health_data[:,9:10], bias=False)) #kurtosis of sex
        print("smoking \t\t",kurtosis(health_data[:,10:11], bias=False)) #kurtosis of smoking
        print("time \t\t\t",kurtosis(health_data[:,11:12], bias=False)) #kurtosis of time
        print("DEATH_EVENT \t\t",kurtosis(health_data[:,-1], bias=False)) #kurtosis of DEATH_EVENT
        
        
    else:
        #print invalid input for non mean or median input
        print('Invalid Input')

#Defining a function to return the measures of spread    
def measures_of_spread(choice):  
    """ function to return measures of spread: std, range and variance """
    
    #Measures of Spread: Standard Deviation
    if choice.lower() == 'std':
        #return np.std(health_data_analysis,axis = 0)##This can be printed to make the code more concise
        print("age \t\t\t",np.std(health_data[:,0:1], ddof=1)) #standard deviation of age
        print("anaemia \t\t",np.std(health_data[:,1:2], ddof=1)) #standard deviation of anaemia
        print("creatinine_phosphokinase",np.std(health_data[:,2:3], ddof=1)) #standard deviation of creatinine_phosphokinase
        print("diabetes \t\t",np.std(health_data[:,3:4], ddof=1)) #standard deviation of diabetes
        print("ejection_fraction \t",np.std(health_data[:,4:5], ddof=1)) #standard deviation of ejection_fraction
        print("high_blood_pressure \t",np.std(health_data[:,5:6], ddof=1)) #standard deviation of high_blood_pressure
        print("platelets \t\t",np.std(health_data[:,6:7], ddof=1)) #standard deviation of platelets
        print("serum_creatinine \t",np.std(health_data[:,7:8], ddof=1)) #standard deviation of serum_creatinine
        print("serum_sodium \t\t",np.std(health_data[:,8:9], ddof=1)) #standard deviation of serum_sodium
        print("sex \t\t\t",np.std(health_data[:,9:10], ddof=1)) #standard deviation of sex
        print("smoking \t\t",np.std(health_data[:,10:11], ddof=1)) #standard deviation of smoking
        print("time \t\t\t",np.std(health_data[:,11:12], ddof=1)) #standard deviation of time
        print("DEATH_EVENT \t\t",np.std(health_data[:,-1], ddof=1)) #standard deviation of DEATH_EVENT
        
    #Measures of Spread: Range
    elif choice.lower() == 'range':    
        print("age \t\t\t",np.max(health_data[:,0:1])- np.min(health_data[:,0:1])) #range of age
        print("anaemia \t\t",np.max(health_data[:,1:2])- np.min(health_data[:,1:2])) #range of anaemia
        print("creatinine_phosphokinase",np.max(health_data[:,2:3])- np.min(health_data[:,2:3])) #range of creatinine_phosphokinase
        print("diabetes \t\t",np.max(health_data[:,3:4])- np.min(health_data[:,3:4])) #range of diabetes
        print("ejection_fraction \t",np.max(health_data[:,4:5])- np.min(health_data[:,4:5])) #range of ejection_fraction
        print("high_blood_pressure \t",np.max(health_data[:,5:6])- np.min(health_data[:,5:6])) #range of high_blood_pressure
        print("platelets \t\t",np.max(health_data[:,6:7])- np.min(health_data[:,6:7])) #range of platelets
        print("serum_creatinine \t",np.max(health_data[:,7:8])- np.min(health_data[:,7:8])) #range of serum_creatinine
        print("serum_sodium \t\t",np.max(health_data[:,8:9])- np.min(health_data[:,8:9])) #range of serum_sodium
        print("sex \t\t\t",np.max(health_data[:,9:10])- np.min(health_data[:,9:10])) #range of sex
        print("smoking \t\t",np.max(health_data[:,10:11])- np.min(health_data[:,10:11])) #range of smoking
        print("time \t\t\t",np.max(health_data[:,11:12])- np.min(health_data[:,11:12])) #range of time
        print("DEATH_EVENT \t\t",np.max(health_data[:,-1])- np.min(health_data[:,-1])) #range of DEATH_EVENT
   
    #Measures of Spread: Variance
    elif choice.lower() == 'var':  
        print("age \t\t\t",np.var(health_data[:,0:1], ddof=1)) #variance of age
        print("anaemia \t\t",np.var(health_data[:,1:2], ddof=1)) #variance of anaemia
        print("creatinine_phosphokinase",np.var(health_data[:,2:3], ddof=1)) #variance of creatinine_phosphokinase
        print("diabetes \t\t",np.var(health_data[:,3:4], ddof=1)) #variance of diabetes
        print("ejection_fraction \t",np.var(health_data[:,4:5], ddof=1)) #variance of ejection_fraction
        print("high_blood_pressure \t",np.var(health_data[:,5:6], ddof=1)) #variance of high_blood_pressure
        print("platelets \t\t",np.var(health_data[:,6:7], ddof=1)) #variance of platelets
        print("serum_creatinine \t",np.var(health_data[:,7:8], ddof=1)) #variance of serum_creatinine
        print("serum_sodium \t\t",np.var(health_data[:,8:9], ddof=1)) #variance of serum_sodium
        print("sex \t\t\t",np.var(health_data[:,9:10], ddof=1)) #variance of sex
        print("smoking \t\t",np.var(health_data[:,10:11], ddof=1)) #variance of smoking
        print("time \t\t\t",np.var(health_data[:,11:12], ddof=1)) #variance of time
        print("DEATH_EVENT \t\t",np.var(health_data[:,-1], ddof=1)) #variance of DEATH_EVENT

    else:
        #print invalid input for non std or range or var input 
        print('Invalid Input')
        
def plots(choice): 
    """ function to plot histogram, barchart, boxplot and heatmap """
    
    #Plots: Histogram
    if choice.lower() == 'hist':
        #distribution plot for age and setting the kde curve to true
        sns.displot(health_data[:,0:1], bins=25, kde=True)
        plt.title('Age Distribution', fontsize=15)
        
        #distribution plot for anaemia
        sns.displot(health_data[:,1:2], bins=10)
        plt.title('Anaemia Distribution', fontsize=15)

        #distribution plot for creatinine_phosphokinase and setting the kde curve to true
        sns.displot(health_data[:,2:3], bins=25, kde = True)
        plt.title('Creatinine Phosphokinase Distribution', fontsize=15)
        
        #distribution plot for diabetes
        sns.displot(health_data[:,3:4], bins=10)
        plt.title('Diabetes Distribution', fontsize=15)
        
        #distribution plot for ejection_fraction and setting the kde curve to true
        sns.displot(health_data[:,4:5], bins=25, kde = True)
        plt.title('Ejection Fraction Distribution', fontsize=15)

        #distribution plot for high_blood_pressure
        sns.displot(health_data[:,5:6], bins=10)
        plt.title('High Blood Pressure Distribution', fontsize=15)
        
        #distribution plot for platelets and setting the kde curve to true
        sns.displot(health_data[:,6:7], bins=25, kde = True)
        plt.title('Platelets Distribution', fontsize=15)

        #distribution plot for serum_creatinine and setting the kde curve to true
        sns.displot(health_data[:,7:8], bins=25, kde = True)
        plt.title('Serum Creatinine Distribution', fontsize=15)

        #distribution plot for serum_sodium and setting the kde curve to true
        sns.displot(health_data[:,8:9], bins=25, kde = True)
        plt.title('Serum Sodium Distribution', fontsize=15)

        #distribution plot for sex
        sns.displot(health_data[:,9:10], bins=10)
        plt.title('Sex Distribution', fontsize=15)

        #distribution plot for smoking
        sns.displot(health_data[:,10:11], bins=10)
        plt.title('Smoking Distribution', fontsize=15)

        #distribution plot for time and setting the kde curve to true
        sns.displot(health_data[:,11:12], bins=25, kde = True)
        plt.title('Time Distribution', fontsize=15)

        #distribution plot for DEATH_EVENT
        sns.displot(health_data[:,-1], bins=10)
        plt.title('DEATH EVENT Distribution', fontsize=15)
        plt.show()
        
    #Plots: BarChart
    elif choice.lower() == 'bar':
        #mapping binary variables
        diabetes_temp= health_data_df['diabetes'].map({0 : 'non-diabetic', 1 : 'diabetic'})#0 = false, 1 = true
        high_blood_pressure_temp = health_data_df['high_blood_pressure'].map({0 : 'no HBP', 1 : 'HBP'})#0 = false, 1 = true
        sex_temp= health_data_df['sex'].map({0 : 'female', 1 : 'male'})#0 = female, 1 = male
        smoking_temp= health_data_df['smoking'].map({0 : 'non-smoker', 1 : 'smoker'})#0 = false, 1 = true
        death_event_temp= health_data_df['DEATH_EVENT'].map({0 : 'survived', 1 : 'dead'})#0 = survived, 1 = dead
    
        #plotting 4 figures and setting the figure size
        fig, axes = plt.subplots(ncols=4, figsize=(20,6))
        
        #Checking death event with diabetes, high_blood_pressure, sex, smoking
        sns.countplot(data = health_data_df, x= diabetes_temp, hue = death_event_temp, ax=axes[0])
        sns.countplot(data=health_data_df, x= high_blood_pressure_temp, hue = death_event_temp, ax = axes[1])
        sns.countplot(data = health_data_df, x= sex_temp, hue = death_event_temp , ax=axes[2])
        sns.countplot(data = health_data_df, x= smoking_temp, hue = death_event_temp , ax=axes[3])
        plt.show()
        
    #Plots: Correlation HeatMap
    #This is used due to the target variable being a binary variable
    elif choice.lower() == 'corr':
        #Setting figure size
        plt.figure(figsize=(12,12))
        
        #Correlation heatmap, setting the range of values from -1 to 1 and the annotation to true to display correlation values
        s=sns.heatmap(health_data_df.corr(), vmin=-1, vmax=1, cmap= 'Blues', annot = True)
        s.set_title('Health Data Correlation Matrix')
        plt.show()
    
    #Plots: Box Plot
    #This can be used to show some insights about the data including outliers
    #Numerical features alone was used
    elif choice.lower() == 'box':
        sns.catplot(data=health_data_df, x="DEATH_EVENT", y="age", kind="box")
        plt.title('age / DEATH_EVENT', fontsize=12)
        
        sns.catplot(data=health_data_df, x="DEATH_EVENT", y="creatinine_phosphokinase", kind="box")
        plt.title('creatinine_phosphokinase / DEATH_EVENT', fontsize=12)
        
        sns.catplot(data=health_data_df, x="DEATH_EVENT", y="ejection_fraction", kind="box")
        plt.title('ejection_fraction / DEATH_EVENT', fontsize=12)
        
        sns.catplot(data=health_data_df, x="DEATH_EVENT", y="platelets", kind="box")
        plt.title('platelets / DEATH_EVENT', fontsize=12)
        
        sns.catplot(data=health_data_df, x="DEATH_EVENT", y="serum_creatinine", kind="box")
        plt.title('serum_creatinine / DEATH_EVENT', fontsize=12)
        
        sns.catplot(data=health_data_df, x="DEATH_EVENT", y="serum_sodium", kind="box")
        plt.title('serum_sodium / DEATH_EVENT', fontsize=12)
        
        sns.catplot(data=health_data_df, x="DEATH_EVENT", y="time", kind="box")
        plt.title('time / DEATH_EVENT', fontsize=12)
        plt.show()
        
    else:
        #print invalid input 
        print('Invalid Input')


# In[69]:


#Extracting the features and the target variables 
X= health_data_df.drop('DEATH_EVENT', axis=1) #features
y = health_data_df['DEATH_EVENT'] #target variable

#setting the target name for the classification report
target_names = ['survived', 'dead']

def classI(choice):
    """ function to perform classification I:
        split dataset into train and test set, 
        feature scaling,
        fit the model,
        evaluate the model - accuracy, precision, recall and F1-Score 
        print confusion matrix """

    #Splitting X and y into training and testing sets, compared 0.20, 0.25 and 0.33, decided to use 0.25 as the test set size for my models
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25) #chose to use the 75:25 train-test split
    
    #Feature scaling, RobustScaler is a preprocessing technique that can be used to handle outliers when building models for supervised training algorithms
    scaler = RobustScaler() #Initialize the scaler    
    X_train = scaler.fit_transform(X_train) #Fit and transform the train set
    X_test = scaler.transform(X_test) #Transform the test set

    #Gaussian Naive Bayes
    if choice.lower() == 'gnb':
        #define the parameter grid
        param_grid = {'var_smoothing': np.logspace(0,-9)}

        #initiate the model
        gnb = GaussianNB()

        #define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #define the GridSearchCV object
        grid_search = GridSearchCV(gnb, param_grid, scoring=scorer, cv=5)

        #fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #get the best estimator from the grid search
        best_gnb = grid_search.best_estimator_

        #fit the best estimator to the training data
        gnb_model = best_gnb.fit(X_train, y_train)

        #make predictions on the test set
        predict_gnb = gnb_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(gnb_model, X_train, y_train)

        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_gnb,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_gnb))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_gnb)
        plt.title('Gaussian Naive Bayes Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(gnb_model, X_train, X_test, y_train, y_test, predict_gnb)
        
        
    #Logistic Regression    
    elif choice.lower() == 'lr':
         #Define the parameter grid
        param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

        #initiate the model
        lr = LogisticRegression()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(lr, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_lr = grid_search.best_estimator_

        #Fit the best estimator to the training data
        lr_model = best_lr.fit(X_train, y_train)

        # Make predictions on the test set
        predict_lr = lr_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(lr_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_lr,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_lr))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_lr)
        plt.title('Logistic Regression Confusion Matrix')
        plt.show()
        
         #Test for overfitting
        test_overfitting(lr_model, X_train, X_test, y_train, y_test, predict_lr)
        
    #Nearest Neighbor Model   
    elif choice.lower() == 'knn':  
        #Define the parameter grid
        param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}

        #initiate the model
        knn = KNeighborsClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(knn, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_knn = grid_search.best_estimator_

        #Fit the best estimator to the training data
        knn_model = best_knn.fit(X_train, y_train)

        #Make predictions on the test set
        predict_knn = knn_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(knn_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_knn,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_knn))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_knn)
        plt.title('Nearest Neighbor Model Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(knn_model, X_train, X_test, y_train, y_test, predict_knn)
     
    #Random Forest Classifier
    elif choice.lower() == 'rfc':  
        #Define the parameter grid
        param_grid = {'n_estimators': [10,50,100], 'bootstrap': [True, False],'criterion': ['gini', 'entropy'], 'max_depth': [None, 3, 5, 10]}
        
        #initiate the model
        rfc = RandomForestClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(rfc, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_rfc = grid_search.best_estimator_

        #Fit the best estimator to the training data
        rfc_model = best_rfc.fit(X_train, y_train)

        #Make predictions on the test set
        predict_rfc = rfc_model.predict(X_test)

        #printing the best parameters
        print("Best parameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)
        
        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(rfc_model, X_train, y_train)
        
        #printing the best parameters
        print("Best parameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)

        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_rfc,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_rfc))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_rfc) 
        plt.title('Random Forest Classifier Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(rfc_model, X_train, X_test, y_train, y_test, predict_rfc)
        
    #Support Vector Machine    
    elif choice.lower() == 'svm': 
        #Define the parameter grid
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

        #initiate the model
        svm = SVC()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(svm, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_svm = grid_search.best_estimator_

        #Fit the best estimator to the training data
        svm_model = best_svm.fit(X_train, y_train)

        #Make predictions on the test set
        predict_svm = svm_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(svm_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_svm,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_svm))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_svm)
        plt.title('Support Vector Machine Confusion Matrix')
        plt.show()

        #Test for overfitting
        test_overfitting(svm_model, X_train, X_test, y_train, y_test, predict_svm)
        
    #Multi-Layer Perceptron Neural Networks
    elif choice.lower() == 'mpnn': 
        #Define the parameter grid
        param_grid = {'hidden_layer_sizes': [(10,), (20,), (30,)], 'alpha': [0.0001, 0.001, 0.01]}

        #initiate the model
        mpnn = MLPClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(mpnn, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_mpnn = grid_search.best_estimator_

        #Fit the best estimator to the training data
        mpmm_model = best_mpnn.fit(X_train, y_train)

        #Make predictions on the test set
        predict_mpnn = mpmm_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(mpmm_model, X_train, y_train)

        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_mpnn,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_mpnn))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_mpnn)
        plt.title('Multi-Layer Perceptron Neural Networks Confusion Matrix')
        plt.show()

        #Test for overfitting
        test_overfitting(mpmm_model, X_train, X_test, y_train, y_test, predict_mpnn)
        
    else:
        #print invalid input 
        print('Invalid Input')

#cross-validation
def cross_validation(model, X_train, y_train):
    """ function to print accuracy to evaluate the model using cross validation """
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print("Accuracy:",round(scores.mean()*100,2),"%")
    print()

#defining the confusion matrix plot to output the confusion matrix for the models        
def confusion_matrix_plot(test, predicted):
    """ function to plot the confusion matrix """
    ConfusionMatrixDisplay.from_predictions(test, predicted, cmap= 'Blues')#plot was set to blue
    plt.title("Confusion Matrix")
    
# function to test for overfitting
def test_overfitting(model, X_train, X_test, y_train, y_test, y_test_pred):
    """ function to test for overfitting """
    # predict on training sets
    y_train_pred = model.predict(X_train)
    
    # calculate mean squared error for training and testing sets
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # if training mse is greater than testing mse, the model is overfitting
    if train_mse > test_mse:
        print("\nMODEL IS OVERFITTING")
    else:
        print("\nMODEL IS NOT OVERFITTING")              


# In[70]:


def class_imb():
    """ function to check for class imbalance in the target variable"""
    #Class Imbalancing
    death_event_temp= health_data_df['DEATH_EVENT'].map({0 : 'survived', 1 : 'dead'})#mapping the class, 0 = survived, 1 = dead
    #The distribution of the values of the target variable
    sns.countplot(death_event_temp)
    plt.title('Imbalance Check')
    plt.show()
    
def smote():
    """ function to perform oversampling (smote) and test its effectiveness """
    #Imbalanced Data Handling Techniques
    #SMOTE 
    # fit predictor and target variable
    X_smote, y_smote = SMOTE().fit_resample(X, y)  
    
    #Class Balancing - SMOTE
    y_balanced = y_smote.value_counts().values
    #The distribution of the values of the target variable
    sns.barplot(['survived','dead'],y_balanced)
    plt.title('Imbalance Check - SMOTE')
    plt.show()

def adasyn():
    """ function to perform oversampling (adasyn) and test its effectiveness """
    #Imbalanced Data Handling Techniques
    #ADASYN
    # fit predictor and target variable
    X_adasyn, y_adasyn = ADASYN().fit_resample(X, y)

    #Class Balancing - ADASYN
    y_balanced =y_adasyn.value_counts().values
    #The distribution of the values of the target variable
    sns.barplot(['survived','dead'],y_balanced)
    plt.title('Imbalance Check - ADASYN')
    plt.show()


def classII_S(choice):   
    """ function to perform classification II using smote:
        split balanced dataset into train and test set, 
        feature scaling,
        perform hyperparameter tuning (gridsearchcv)
        fit the model,
        evaluate the model - accuracy, precision, recall and F1-Score 
        test for overfitting
        print confusion matrix """
    
    #SMOTE imbalanced Data Handling Technique
    # fit predictor and target variable
    X_smote, y_smote = SMOTE().fit_resample(X, y)

    #Splitting X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_smote,y_smote,test_size=0.25) #75:25 train-test split
    
    #Feature scaling, it also handles outliers
    scaler = RobustScaler() #Initialize the scaler    
    X_train = scaler.fit_transform(X_train) #Fit and transform the train set
    X_test = scaler.transform(X_test) #Transform the test set
    
    #Gaussian Naive Bayes
    if choice.lower() == 'gnb':
        #define the parameter grid
        param_grid = {'var_smoothing': np.logspace(0,-9)}

        #initiate the model
        gnb = GaussianNB()

        #define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #define the GridSearchCV object
        grid_search = GridSearchCV(gnb, param_grid, scoring=scorer, cv=5)

        #fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #get the best estimator from the grid search
        best_gnb = grid_search.best_estimator_

        #fit the best estimator to the training data
        gnb_model = best_gnb.fit(X_train, y_train)

        #make predictions on the test set
        predict_gnb = gnb_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(gnb_model, X_train, y_train)

        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_gnb,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_gnb))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_gnb)
        plt.title('Gaussian Naive Bayes Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(gnb_model, X_train, X_test, y_train, y_test, predict_gnb)
        
        
    #Logistic Regression    
    elif choice.lower() == 'lr':
         #Define the parameter grid
        param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

        #initiate the model
        lr = LogisticRegression()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(lr, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_lr = grid_search.best_estimator_

        #Fit the best estimator to the training data
        lr_model = best_lr.fit(X_train, y_train)

        # Make predictions on the test set
        predict_lr = lr_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(lr_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_lr,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_lr))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_lr)
        plt.title('Logistic Regression Confusion Matrix')
        plt.show()
        
         #Test for overfitting
        test_overfitting(lr_model, X_train, X_test, y_train, y_test, predict_lr)
        
    #Nearest Neighbor Model   
    elif choice.lower() == 'knn':  
        #Define the parameter grid
        param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}

        #initiate the model
        knn = KNeighborsClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(knn, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_knn = grid_search.best_estimator_

        #Fit the best estimator to the training data
        knn_model = best_knn.fit(X_train, y_train)

        #Make predictions on the test set
        predict_knn = knn_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(knn_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_knn,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_knn))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_knn)
        plt.title('Nearest Neighbor Model Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(knn_model, X_train, X_test, y_train, y_test, predict_knn)
     
    #Random Forest Classifier
    elif choice.lower() == 'rfc':  
        #Define the parameter grid
        param_grid = {'n_estimators': [10,50,100], 'bootstrap': [True, False],'criterion': ['gini', 'entropy'], 'max_depth': [None, 3, 5, 10]}
    
        #initiate the model
        rfc = RandomForestClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(rfc, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_rfc = grid_search.best_estimator_

        #Fit the best estimator to the training data
        rfc_model = best_rfc.fit(X_train, y_train)

        #Make predictions on the test set
        predict_rfc = rfc_model.predict(X_test)

        #printing the best parameters
        print("Best parameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)
        
        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(rfc_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_rfc,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_rfc))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_rfc) 
        plt.title('Random Forest Classifier Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(rfc_model, X_train, X_test, y_train, y_test, predict_rfc)
        
    #Support Vector Machine    
    elif choice.lower() == 'svm': 
        #Define the parameter grid
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

        #initiate the model
        svm = SVC()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(svm, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_svm = grid_search.best_estimator_

        #Fit the best estimator to the training data
        svm_model = best_svm.fit(X_train, y_train)

        #Make predictions on the test set
        predict_svm = svm_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(svm_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_svm,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_svm))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_svm)
        plt.title('Support Vector Machine Confusion Matrix')
        plt.show()

        #Test for overfitting
        test_overfitting(svm_model, X_train, X_test, y_train, y_test, predict_svm)
        
    #Multi-Layer Perceptron Neural Networks
    elif choice.lower() == 'mpnn': 
        #Define the parameter grid
        param_grid = {'hidden_layer_sizes': [(10,), (20,), (30,)], 'alpha': [0.0001, 0.001, 0.01]}

        #initiate the model
        mpnn = MLPClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(mpnn, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_mpnn = grid_search.best_estimator_

        #Fit the best estimator to the training data
        mpmm_model = best_mpnn.fit(X_train, y_train)

        #Make predictions on the test set
        predict_mpnn = mpmm_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(mpmm_model, X_train, y_train)

        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_mpnn,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_mpnn))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_mpnn)
        plt.title('Multi-Layer Perceptron Neural Networks Confusion Matrix')
        plt.show()

        #Test for overfitting
        test_overfitting(mpmm_model, X_train, X_test, y_train, y_test, predict_mpnn)
        
    else:
        #print invalid input 
        print('Invalid Input')    
        
def classII_A(choice): 
    """ function to perform classification II using adasyn:
        split balanced dataset into train and test set, 
        feature scaling,
        perform hyperparameter tuning (gridsearchcv)
        fit the model,
        evaluate the model - accuracy, precision, recall and F1-Score 
        test for overfitting
        print confusion matrix """
    
    #ADASYN imbalanced Data Handling Technique
    # fit predictor and target variable
    X_adasyn, y_adasyn = ADASYN().fit_resample(X, y)
    
    #Splitting X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_adasyn,y_adasyn,test_size=0.25) #chose to use the 75:25 train-test split

    #Feature scaling, it also handles outliers
    scaler = RobustScaler() #Initialize the scaler    
    X_train = scaler.fit_transform(X_train) #Fit and transform the train set
    X_test = scaler.transform(X_test) #Transform the test set
    
    #Gaussian Naive Bayes
    if choice.lower() == 'gnb':
        #Define the parameter grid
        param_grid = {'var_smoothing': np.logspace(0,-9)}

        #initiate the model
        gnb = GaussianNB()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(gnb, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_gnb = grid_search.best_estimator_

        #Fit the best estimator to the training data
        gnb_model = best_gnb.fit(X_train, y_train)

        #Make predictions on the test set
        predict_gnb = gnb_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(gnb_model, X_train, y_train)

        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_gnb,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_gnb))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_gnb)
        plt.title('Gaussian Naive Bayes Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(gnb_model, X_train, X_test, y_train, y_test, predict_gnb)
        
        
    #Logistic Regression    
    elif choice.lower() == 'lr':
         #Define the parameter grid
        param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

        #initiate the model
        lr = LogisticRegression()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(lr, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_lr = grid_search.best_estimator_

        #Fit the best estimator to the training data
        lr_model = best_lr.fit(X_train, y_train)

        #Make predictions on the test set
        predict_lr = lr_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(lr_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_lr,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_lr))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_lr)
        plt.title('Logistic Regression Confusion Matrix')
        plt.show()
        
         #Test for overfitting
        test_overfitting(lr_model, X_train, X_test, y_train, y_test, predict_lr)
        
     #Nearest Neighbor Model   
    elif choice.lower() == 'knn':  
        #Define the parameter grid
        param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}

        #initiate the model
        knn = KNeighborsClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(knn, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_knn = grid_search.best_estimator_

        #Fit the best estimator to the training data
        knn_model = best_knn.fit(X_train, y_train)

        #Make predictions on the test set
        predict_knn = knn_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(knn_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_knn,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_knn))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_knn)
        plt.title('Nearest Neighbor Model Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(knn_model, X_train, X_test, y_train, y_test, predict_knn)
     
    #Random Forest Classifier
    elif choice.lower() == 'rfc':  
        #Define the parameter grid
        param_grid = {'n_estimators': [10,50,100], 'bootstrap': [True, False],'criterion': ['gini', 'entropy'], 'max_depth': [None, 3, 5, 10]}
    
        #initiate the model
        rfc = RandomForestClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(rfc, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_rfc = grid_search.best_estimator_

        #Fit the best estimator to the training data
        rfc_model = best_rfc.fit(X_train, y_train)

        #Make predictions on the test set
        predict_rfc = rfc_model.predict(X_test)

        #printing the best parameters
        print("Best parameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)
        
        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(rfc_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_rfc,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_rfc))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_rfc) 
        plt.title('Random Forest Classifier Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(rfc_model, X_train, X_test, y_train, y_test, predict_rfc)
        
    #Support Vector Machine    
    elif choice.lower() == 'svm': 
        #Define the parameter grid
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

        #initiate the model
        svm = SVC()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(svm, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_svm = grid_search.best_estimator_

        #Fit the best estimator to the training data
        svm_model = best_svm.fit(X_train, y_train)

        #Make predictions on the test set
        predict_svm = svm_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(svm_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_svm,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_svm))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_svm)
        plt.title('Support Vector Machine Confusion Matrix')
        plt.show()

        #Test for overfitting
        test_overfitting(svm_model, X_train, X_test, y_train, y_test, predict_svm)
        
    #Multi-Layer Perceptron Neural Networks
    elif choice.lower() == 'mpnn': 
        #Define the parameter grid
        param_grid = {'hidden_layer_sizes': [(10,), (20,), (30,)], 'alpha': [0.0001, 0.001, 0.01]}

        #initiate the model
        mpnn = MLPClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(mpnn, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_mpnn = grid_search.best_estimator_

        #Fit the best estimator to the training data
        mpmm_model = best_mpnn.fit(X_train, y_train)

        #Make predictions on the test set
        predict_mpnn = mpmm_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(mpmm_model, X_train, y_train)

        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_mpnn,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_mpnn))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_mpnn)
        plt.title('Multi-Layer Perceptron Neural Networks Confusion Matrix')
        plt.show()

        #Test for overfitting
        test_overfitting(mpmm_model, X_train, X_test, y_train, y_test, predict_mpnn)
        
    else:
        #print invalid input 
        print('Invalid Input')            


# In[71]:


def feature_selection(choice):
    """ function for feature selection using mann-whitney test, chi square test and random forest importance plot"""
        
    if choice.lower() == 'm_w':
        #Mann-Whitney test for feature selection
        
        X= health_data_df.drop('DEATH_EVENT', axis=1) #features
        y = health_data_df['DEATH_EVENT'] #target variable
        
        #get the column names of the features
        column = X.columns
        
        #set threshold for p-values
        threshold = 0.05

        #empty list to store the pvalue
        pval = []

        #iterate through the features
        for i in column:
            # calculate the statistic and pvalue for each feature
            statistic, pvalue = mannwhitneyu(health_data_df[i][health_data_df['DEATH_EVENT']==1], health_data_df[i][health_data_df['DEATH_EVENT']==0], alternative='two-sided')
            pval.append(pvalue)

        # convert the list of p-values to a dataframe    
        pvalue_dataFrame = pd.DataFrame(pval,columns = ['Pvalue'],index = column)

        # check the pvalue against the threshold
        pvalue_dataFrame['result'] = 'Accept H0'
        pvalue_dataFrame.loc[pvalue_dataFrame.Pvalue <= threshold, 'result'] = 'Reject H0'

        print(pvalue_dataFrame)

        #bar plot to visualize the feature importance 
        pvalue_dataFrame.set_index(X.columns).sort_values(by = ['Pvalue']).plot(kind="barh") #set the index to the feature and sort the pvalue
        plt.xlabel("p_values",fontsize=20)
        plt.ylabel("Features",fontsize=20)
        plt.title("Mann-Whitney Test")
        plt.show()
        
    elif choice.lower() == 'chi':
        #Chi-square test for feature selection
        
        X= health_data_df.drop('DEATH_EVENT', axis=1) #features
        y = health_data_df['DEATH_EVENT'] #target variable
        
        # set threshold for p-values
        threshold = 0.05

        # calculate the chi-square statistic and p-value for each feature
        chi_value, pvalue = chi2(X, y)

        # convert the p-values to a dataframe
        pvalue_dataFrame = pd.DataFrame(pvalue,columns = ['Pvalue'],index = X.columns)

        # check the pvalue against the threshold
        pvalue_dataFrame['result'] = 'Accept H0'
        pvalue_dataFrame.loc[pvalue_dataFrame.Pvalue <= threshold, 'result'] = 'Reject H0'

        print(pvalue_dataFrame)

        # bar plot to visualize the feature importance 
        pvalue_dataFrame.sort_values(by = 'Pvalue').plot(kind="barh",y='Pvalue')
        plt.xlabel("p_values",fontsize=15)
        plt.ylabel("Features",fontsize=15)
        plt.title("Chi-Square Test")
        plt.show()

        
    elif choice.lower() == 'rfc':
        X= health_data_df.drop('DEATH_EVENT', axis=1) #features
        y = health_data_df['DEATH_EVENT'] #target variable

        #Splitting X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25) #75:25 train-test split

        #Feature scaling, it also handles outliers
        scaler = RobustScaler() #Initialize the scaler    
        X_train = scaler.fit_transform(X_train) #Fit and transform the train set
        X_test = scaler.transform(X_test) #Transform the test set

        # Random Forest feature importance plot with GridSearchCV

        # Define the parameter grid
        param_grid = {'n_estimators': [10,50,100], 'bootstrap': [True, False],'criterion': ['gini', 'entropy'], 'max_depth': [None, 3, 5, 10]}
    
        # initiate the model
        rfc = RandomForestClassifier()

        # Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        # Define the GridSearchCV object
        grid_search = GridSearchCV(rfc, param_grid, scoring=scorer, cv=5)

        # Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        # Get the best estimator from the grid search
        best_rfc = grid_search.best_estimator_

        # Fit the best estimator to the training data
        rfc_model = best_rfc.fit(X_train, y_train)

        # Make predictions on the test set
        predict_rfc = rfc_model.predict(X_test)

        importance = rfc_model.feature_importances_
        features_imp = pd.DataFrame({'Features': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending = False)

        # feature importance plot
        features_imp.set_index('Features', inplace=True, drop=True)#to set index to features
        fig = features_imp.plot(kind="barh", color = 'blue')
        fig.set_title('RFC Feature Importance Plot')
        fig.set_xlabel("Feature Importance")
        fig.set_ylabel("Features")
        plt.show()
        
    else:
        #print invalid input 
        print('Invalid Input')  


# In[88]:


#Extracting the important features and the target variable
X= health_data_df.drop(columns=['sex', 'platelets', 'anaemia', 'creatinine_phosphokinase', 'high_blood_pressure','smoking', 'diabetes', 'serum_sodium', 'DEATH_EVENT']) #selected features
y = health_data_df['DEATH_EVENT'] #target variable

def classIII(choice):
    """ function to perform classification III using smote:
        extract important features and target variable
        split balanced dataset into train and test set, 
        feature scaling,
        perform hyperparameter tuning (gridsearchcv)
        fit the model,
        evaluate the model - accuracy, precision, recall and F1-Score 
        test for overfitting
        print confusion matrix """
    
    #Splitting X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25) #75:25 train-test split

    #Feature scaling, it also handles outliers
    scaler = RobustScaler() #Initialize the scaler    
    X_train = scaler.fit_transform(X_train) #Fit and transform the train set
    X_test = scaler.transform(X_test) #Transform the test set
    
    #Gaussian Naive Bayes
    if choice.lower() == 'gnb':
        #define the parameter grid
        param_grid = {'var_smoothing': np.logspace(0,-9)}

        #initiate the model
        gnb = GaussianNB()

        #define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #define the GridSearchCV object
        grid_search = GridSearchCV(gnb, param_grid, scoring=scorer, cv=5)

        #fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #get the best estimator from the grid search
        best_gnb = grid_search.best_estimator_

        #fit the best estimator to the training data
        gnb_model = best_gnb.fit(X_train, y_train)

        #make predictions on the test set
        predict_gnb = gnb_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(gnb_model, X_train, y_train)

        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_gnb,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_gnb))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_gnb)
        plt.title('Gaussian Naive Bayes Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(gnb_model, X_train, X_test, y_train, y_test, predict_gnb)
        
        
    #Logistic Regression    
    elif choice.lower() == 'lr':
         #Define the parameter grid
        param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

        #initiate the model
        lr = LogisticRegression()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(lr, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_lr = grid_search.best_estimator_

        #Fit the best estimator to the training data
        lr_model = best_lr.fit(X_train, y_train)

        # Make predictions on the test set
        predict_lr = lr_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(lr_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_lr,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_lr))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_lr)
        plt.title('Logistic Regression Confusion Matrix')
        plt.show()
        
         #Test for overfitting
        test_overfitting(lr_model, X_train, X_test, y_train, y_test, predict_lr)
        
    #Nearest Neighbor Model   
    elif choice.lower() == 'knn':  
        #Define the parameter grid
        param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}

        #initiate the model
        knn = KNeighborsClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(knn, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_knn = grid_search.best_estimator_

        #Fit the best estimator to the training data
        knn_model = best_knn.fit(X_train, y_train)

        #Make predictions on the test set
        predict_knn = knn_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(knn_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_knn,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_knn))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_knn)
        plt.title('Nearest Neighbor Model Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(knn_model, X_train, X_test, y_train, y_test, predict_knn)
     
    #Random Forest Classifier
    elif choice.lower() == 'rfc':  
        #Define the parameter grid
        param_grid = {'n_estimators': [10,50,100], 'bootstrap': [True, False],'criterion': ['gini', 'entropy'], 'max_depth': [None, 3, 5, 10]}

        #initiate the model
        rfc = RandomForestClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(rfc, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_rfc = grid_search.best_estimator_

        #Fit the best estimator to the training data
        rfc_model = best_rfc.fit(X_train, y_train)

        #Make predictions on the test set
        predict_rfc = rfc_model.predict(X_test)
        
        #printing the best parameters
        print("Best parameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)
        
        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(rfc_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_rfc,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_rfc))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_rfc) 
        plt.title('Random Forest Classifier Confusion Matrix')
        plt.show()
        
        #Test for overfitting
        test_overfitting(rfc_model, X_train, X_test, y_train, y_test, predict_rfc)
        
    #Support Vector Machine    
    elif choice.lower() == 'svm': 
        #Define the parameter grid
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

        #initiate the model
        svm = SVC()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(svm, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_svm = grid_search.best_estimator_

        #Fit the best estimator to the training data
        svm_model = best_svm.fit(X_train, y_train)

        #Make predictions on the test set
        predict_svm = svm_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(svm_model, X_train, y_train)
        
        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_svm,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_svm))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_svm)
        plt.title('Support Vector Machine Confusion Matrix')
        plt.show()

        #Test for overfitting
        test_overfitting(svm_model, X_train, X_test, y_train, y_test, predict_svm)
        
    #Multi-Layer Perceptron Neural Networks
    elif choice.lower() == 'mpnn': 
        #Define the parameter grid
        param_grid = {'hidden_layer_sizes': [(10,), (20,), (30,)], 'alpha': [0.0001, 0.001, 0.01]}

        #initiate the model
        mpnn = MLPClassifier()

        #Define the scoring method
        scorer = make_scorer(balanced_accuracy_score)

        #Define the GridSearchCV object
        grid_search = GridSearchCV(mpnn, param_grid, scoring=scorer, cv=5)

        #Fit the GridSearchCV object to the data
        grid_search.fit(X_train, y_train)

        #Get the best estimator from the grid search
        best_mpnn = grid_search.best_estimator_

        #Fit the best estimator to the training data
        mpmm_model = best_mpnn.fit(X_train, y_train)

        #Make predictions on the test set
        predict_mpnn = mpmm_model.predict(X_test)

        #Model Evaluation using Cross Validation
        #Accuracy score
        cross_validation(mpmm_model, X_train, y_train)

        #classification report
        print("CLASSIFICATION REPORT\n")
        print(classification_report(y_test,predict_mpnn,target_names = target_names))

        #Confusion matrix
        print("CONFUSION MATRIX\n")
        print(confusion_matrix(y_test,predict_mpnn))

        #Confusion_matrix_plot function
        confusion_matrix_plot(y_test, predict_mpnn)
        plt.title('Multi-Layer Perceptron Neural Networks Confusion Matrix')
        plt.show()

        #Test for overfitting
        test_overfitting(mpmm_model, X_train, X_test, y_train, y_test, predict_mpnn)
        
    else:
        #print invalid input 
        print('Invalid Input')   

