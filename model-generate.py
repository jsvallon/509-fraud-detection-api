import pandas as pd
     #2---Load and Explore the Dataset
#Upload the path to your actual dataset location
file_path = "/content/creditcard_for_fraud_detection.csv"

#Load the dataset
df = pd.read_csv(file_path)
#Check if it loaded correctly
print(df.head())
print(df.info())

 #3 Data Processing
  #3.1 Handle Imbalanced Data : Fraud cases are rare. so we use SMOTE(Synthetic Minority Oversampling):
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#Separate features and target
X = df.drop(columns=["Class"]) #Features
y = df["Class"] #Target (0 = Not Fraud, 1 = Fraud)

#Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy=0.5, random_state=42)
print(y_train.isnull().sum())  # Count NaN values in target
# Drop rows where the target 'Class' is NaN
X_train = X_train[y_train.notna()]
y_train = y_train.dropna()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
  #3.2 Scale Features : Since the dataset contains large numerical values, we normalize them

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

     #4 Train a Machine Learning Model
#We will start with Random Forest
      #4.1 Train the Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


#Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

#Make predictions
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
#Evaluation performance
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

#Save the trained model
import joblib
#joblib.dump(model, "/content/fraud_model.pkl")
#print("Model saved successfully!")

       #5 Visualize Data(Optional but Recommended)
#The understand fraud vs non-fraud distributions
import matplotlib.pyplot as plt
import seaborn as sns

#Count fraud vs non-fraud
sns.countplot(x=df["Class"])
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()