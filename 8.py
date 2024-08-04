from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

for x, actual in zip(X_test, y_test):
    prediction = kn.predict([x])
    print(f"\nActual:[{actual}][{iris.target_names[actual]}], Predicted:{prediction[0]}[{iris.target_names[prediction][0]}]")

# Display test score
print(f"\nTEST SCORE[ACCURACY]: {kn.score(X_test, y_test)}")