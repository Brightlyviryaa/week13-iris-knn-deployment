import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# random seed
seed = 42

# read original dataset
iris_df = pd.read_csv("Iris.csv")
iris_df = iris_df.sample(frac=1, random_state=seed)  # shuffle for good measure

# selecting features and target data
X = iris_df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = iris_df[["Species"]]

# split data into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=seed)

# train the classifier on the training data
clf.fit(X_train, y_train.values.ravel())

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # expected around 0.91

# save the model to disk
joblib.dump(clf, "rf_model.sav")
