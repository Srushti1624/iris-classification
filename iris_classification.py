import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# STEP 1: Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target # Add the species column to the DataFrame

print("Dataset Loaded Successfully")
print(df.head())
print("Shape:", df.shape)

# STEP 2: Explore visually (scatter plots)
for i, species in enumerate(iris.target_names):
    plt.scatter(
        df[df['species'] == i]['petal length (cm)'],
        df[df['species'] == i]['petal width (cm)'],
        label=species
    )

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.show()

# STEP 3: Split into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[iris.feature_names], df['species'], test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# STEP 4: Preprocess — dataset is already clean, no action needed
print("Missing values:", df.isnull().sum().sum())   # will print 0

# STEP 5: Train classifiers (Logistic Regression)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Model trained")

# STEP 6: Evaluate — accuracy, precision, confusion matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Predict
y_pred = model.predict(X_test)

# Accuracy & Precision
print("Accuracy :", round(accuracy_score(y_test, y_pred), 2))
print("Precision:", round(precision_score(y_test, y_pred, average='macro'), 2))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
).plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()
