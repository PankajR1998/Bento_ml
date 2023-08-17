import bentoml
from sklearn import svm
from sklearn import datasets

# load iris dataset.

iris = datasets.load_iris()
X, y = iris.data , iris.target

# train the model 
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# save model to the Bento Ml local model store 
saved_model = bentoml.sklearn.save_model("iris_clf", clf)

print(f"Model Saved: {saved_model}")

# all the model will be stored in Local machine C-drive.