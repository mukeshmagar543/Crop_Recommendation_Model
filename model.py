# Import Data Manipualtion Libraries
import pandas as pd
import numpy as np

# Imoport Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import Machine Learning Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import Warnings Library
import warnings
warnings.filterwarnings('ignore')

# Impoer Logging Files
import logging
logging.basicConfig(level= logging.INFO,
                    filename='log.txt',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s'

)

# Load the dataset from the given URL
url = 'https://raw.githubusercontent.com/mukeshmagar543/Crop_Recommendation_Model/refs/heads/main/Crop_Recommendation.csv'

df = pd.read_csv(url)

# Converting Categorical column  to Numerical ---> Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Crop'] = le.fit_transform(df['Crop'])

# Splitting the dataset into features (X) and target (y) variables

X = df.drop(columns=['Crop','Rainfall'])

y = df['Crop']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()

RF.fit(X_train, y_train)

y_pred_RF = RF.predict(X_test)

accuracy_score_RF = accuracy_score(y_test, y_pred_RF)

print(accuracy_score_RF)