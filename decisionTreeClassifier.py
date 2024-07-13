import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'D:\\Data Analyst\\Prodigy InfoTech\\Task-3\\cleaned_bank_data.xlsx'
data = pd.read_excel(file_path)

# Drop rows with missing values
data = data.dropna()

# Initialize LabelEncoder
le = LabelEncoder()

# List of categorical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# Apply LabelEncoder to each categorical column
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Convert the target variable 'y' to binary
data['y'] = le.fit_transform(data['y'])

# Define feature columns and target column
feature_columns = data.columns[:-1]
target_column = 'y'

X = data[feature_columns]
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier with max depth
clf = DecisionTreeClassifier(max_depth=5)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=feature_columns, class_names=['no', 'yes'], fontsize=10)
plt.title("Decision Tree Classifier - Customer Purchase Prediction", fontsize=15)
plt.show()
