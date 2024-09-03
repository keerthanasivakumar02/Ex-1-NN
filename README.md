<H3>ENTER YOUR NAME : KEERTHANA S</H3>
<H3>ENTER YOUR REGISTER NO : 212223040092</H3>
<H3>EX. NO.1</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('/content/Churn_Modelling.csv')
print(df.head())

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())

df.duplicated()

df.describe()

df = df.drop(['Surname', 'Geography','Gender'], axis=1)
df.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))

```







## OUTPUT:
## Dataset:
![image](https://github.com/user-attachments/assets/77a51548-d99e-48fd-b8a0-bf618d0719d1)

## X Values:

![image](https://github.com/user-attachments/assets/e99717ce-4d82-486a-9e07-b2bb845a1dfe)


## Y Values:

![image](https://github.com/user-attachments/assets/7f609aff-3061-4536-bf44-d749a00638ce)


## Null Values:

![image](https://github.com/user-attachments/assets/790ef1d4-ac93-446e-a17c-0d154479861d)


## Duplicated Values:

![image](https://github.com/user-attachments/assets/079797ef-b8dc-445b-97de-30686fb473c8)

## Description:

![image](https://github.com/user-attachments/assets/9a2f71c7-fe31-4c44-acff-0e35f8a1fbb8)

## Normalized Dataset:

![image](https://github.com/user-attachments/assets/13a1911f-4b69-43cd-abfd-2a345aa8f6c4)

![image](https://github.com/user-attachments/assets/17e59073-e591-4a16-8d5a-04507686ef08)


## Training Dataset:

![image](https://github.com/user-attachments/assets/df02a704-704d-47de-8178-6b6d0eda9f9e)

## Testing Dataset:

![image](https://github.com/user-attachments/assets/4866fafc-aa53-41b3-9d7d-ee904258ff11)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


