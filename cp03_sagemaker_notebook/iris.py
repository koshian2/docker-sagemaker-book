from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    X_all, y_all = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=123)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)

if __name__ == "__main__":
    main()