# Email Spam Classifier Project

# Importing libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Loading dataset
df = pd.read_csv(
    r"C:\Users\ADMIN\OneDrive\Desktop\AI-Internship\AI-INTERN\spam.csv",
    encoding='latin-1'
)

# Keeping only required columns
df = df[['v1', 'v2']]

# Renaming columns
df.columns = ['label', 'message']

# Display first 5 rows
print("\nFIRST 5 ROWS:\n")
print(df.head())

# Convert labels into numbers
# spam = 1
# ham = 0

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Splitting features and target
X = df['message']
y = df['label']

# Convert text into numeric vectors
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:\n")
print(f"{accuracy * 100:.2f}%")

# Classification report
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

# Testing custom message
sample_message = ["Congratulations! You won a free lottery ticket"]

sample_vector = vectorizer.transform(sample_message)

prediction = model.predict(sample_vector)

print("\nCUSTOM MESSAGE TEST:\n")

if prediction[0] == 1:
    print("SPAM MESSAGE")
else:
    print("NOT SPAM")

# End of project