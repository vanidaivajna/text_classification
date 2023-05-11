import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('your_data.csv')  # Replace 'your_data.csv' with your actual file path

# Assuming 'your_column_name' contains the target labels
X = data.drop('your_column_name', axis=1)
y = data['your_column_name']

# Splitting the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Further split the remaining 80% (X_train, y_train) into train and validation sets (60% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

# Print the shapes of the resulting datasets
print("Train set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)
