<H3>Name: Malar Mariam S</H3>
<H3>Register No: 212223230118</H3>
<H3>EX.NO - 1</H3>
<!-- <H3>Date</H3> -->
<H1 ALIGN =CENTER>Introduction to Kaggle and Data preprocessing</H1>

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
#### STEP 1:
Importing the libraries<BR>
#### STEP 2:
Importing the dataset<BR>
#### STEP 3:
Taking care of missing data<BR>
#### STEP 4:
Encoding categorical data<BR>
#### STEP 5:
Normalizing the data<BR>
#### STEP 6:
Splitting the data into test and train<BR>

##  PROGRAM:

```py
# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Step 2: Load Dataset
df = pd.read_csv("/kaggle/input/iris/Iris.csv")
print(df.head())

# Step 3: Check for Missing Data
print(df.isnull().sum())

# Step 4: encode categorical data
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])
df['Species'] = df['Species'].astype(str).map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})

# step 5: normalizing the feature columns
scaler = StandardScaler()
df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = scaler.fit_transform(
    df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
)

# Step 6: Splitting the data into train and test sets
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Training Target Shape: {y_train.shape}")
print(f"Testing Target Shape: {y_test.shape}")
```

## OUTPUT:
#### dataset
![image](https://github.com/user-attachments/assets/bd1d9539-e3ff-4f67-87f4-273eda96ed2f)

#### missing values
![image](https://github.com/user-attachments/assets/d50baade-e697-4e81-a449-169a3b975ac1)

#### encoded categorical data
![image](https://github.com/user-attachments/assets/86afb306-6ba2-46ed-a1ef-07833206607a)

![image](https://github.com/user-attachments/assets/3f16f63e-43b0-43b8-a661-275f7e81276c)

#### normalised data
![image](https://github.com/user-attachments/assets/4fccc35e-2d36-416d-b2aa-67c7c1c6240e)

#### test, train data
![image](https://github.com/user-attachments/assets/69336e39-d202-4b76-9112-8e5501d5c8b8)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


