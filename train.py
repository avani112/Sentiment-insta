from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from load_data import load_train_test


def fit_naivebayes(X_train, y_train, X_test, y_test, batch=10000):
    clf = MultinomialNB()
    for i in range(0, X_train.shape[0], batch):
        clf.partial_fit(X_train[i:i + batch, :], y_train[i:i + batch], classes=[0, 1])
    yy_test = clf.predict(X_test)
    print('Accuracy:{}'.format(accuracy_score(y_test, yy_test)))
    return clf

if __name__ == "__main__":
    print("Loading data")
    train_X, test_X, train_y, test_y = load_train_test()
    print("Extracting features(TFDF)")
    tfdf = TfidfVectorizer(analyzer="word")
    tfdf.fit(train_X) 
    X_train_dtm = tfdf.transform(train_X)  
    X_test_dtm = tfdf.transform(test_X)  
    train_X, test_X = None, None 

    print("Training logistic regression")
    current_clf = fit_naivebayes(X_train_dtm, train_y, X_test_dtm, test_y)

    print("Saving Artifacts")
    joblib.dump(dict(clf=current_clf, tfdf=tfdf), open("./cache/predict_param.pickle", "wb"))
