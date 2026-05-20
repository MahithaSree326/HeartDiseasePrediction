import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("heart.csv")

# Display first 5 rows
print(data.head())

# Display dataset size
print("\nDataset Shape:")
print(data.shape)

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())


# Count plot for target variable
sns.countplot(x='target', data=data)

# Title
plt.title("Heart Disease Count")

# Show graph without stopping program
plt.show(block=False)


# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define input features and target
X = data.drop('target', axis=1)
y = data['target']

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print("\nModel Accuracy:")
print(accuracy)


from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Labels and title
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Show graph
plt.show()



