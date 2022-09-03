import argparse
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main(opt):
    X_all, y_all = fetch_covtype(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state=123, test_size=0.2)
    print("train: ", X_train.shape)
    print("test: ", X_test.shape)

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    if opt.classifier == "logistic":
        classifier = LogisticRegression()
        classifier.fit(X_train_norm, y_train)
        y_pred = classifier.predict(X_test_norm)
    elif opt.classifier == "naive_bayes":
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    elif opt.classifier == "random_forest":
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    elif opt.classifier == "tree":
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

    print(opt.classifier, accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine multiple classifier")
    parser.add_argument("--classifier", type=str, default="knn")

    opt = parser.parse_args()

    main(opt)