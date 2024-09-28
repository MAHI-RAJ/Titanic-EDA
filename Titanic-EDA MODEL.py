#Let's import some libraries to get started!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Let's start by reading in the titanic_train.csv file into a pandas dataframe

train=pd.read_csv('titanic_train.csv')
train.head()

#let's begin some exploratory data analysis! We'll start by checking out missing data!

train.isnull() # gives values such as True or False!

#Iterating through thousands of values is impossible , so here's a substitute of it in graphical format !

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Let' see how many counts survived & divide them on basis of sex , through seaborn Counterplot !

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train) 

# Now let's find out how many counts survived on basis of passanger classes , *1st class is the richest* ! 

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

#Let's draw a bar chart (HISTOGRAM) showing the count of people of certain age !
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)

#We can perform the above task using matplotlib as well !
train['Age'].hist(bins=30,color='darkred',alpha=0.3)

#Let's see how many counts had any sibling or spouse !
sns.countplot(x='SibSp',data=train)

#A graph on Fare of the tickets people bought !
train['Fare'].hist(color='green',bins=40,figsize=(8,4))

##DATA CLEANING**-----------------------------------------------------------

#Checking the average age by passenger class !
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

#We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these above output of average age values to impute based on Pclass for Age !
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37 # As avg age value for Passenger class 1 is: 37 *from above output !  

        elif Pclass == 2:
            return 29  #As avg age value for Passenger class 2 is: 29

        else:
            return 24  #As avg age value for passenger class 3 is: 24

    else:
        return Age

#Now let's apply the above def function !
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#Now let's check that heat map again!
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Now let's go ahead and drop the Cabin column and the row in Embarked that is NaN!!
train.drop('Cabin',axis=1,inplace=True)

#Let's check , it's deleted or not !!
train.head()

##CONVERTING CATEGORICAL FEATURES !!--------------------------------------------

##Let's convert cetagorical features to dummy variables using pandas !!

pd.get_dummies(train['Embarked'],drop_first=True).head()

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

#Let's drop few columns !!

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()
#Our data is now ready for our model !

##BUILDING A LOGISTIC REGRESSION MODEL !!

#Let's try Train Test Split 
train.drop('Survived',axis=1).head()

train['Survived'].head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.30, random_state=101)

##TRAINING AND PREDICTING !!

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y_test,predictions)

accuracy

from sklearn.metrics import accuracy_score

accuracy

predictions

##let's Evaluate our Model  !!

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))






 




 


