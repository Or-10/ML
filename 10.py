from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

iris = datasets.load_iris()
X= iris.data[:,:2]
y = iris.target

X_train, X_test ,y_train,y_test = train_test_split(X , y ,test_size=0.2,random_state=42)

svm =SVC(kernel='linear', C=0.1,random_state=42)
svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)

print("accuracy:" , accuracy_score(y_test,y_pred))
print("classification report:" , classification_report(y_test,y_pred))
