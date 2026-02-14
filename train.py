import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def parse_features(text):
    feature_list = []
    rows = text.strip().split("\n")
    for row in rows:
        if not row.strip():
            continue
        vals = list(map(float, row.strip().split()))
        feature_list.append(vals)
    return np.array(feature_list)


def parse_labels(text):
    labels = list(map(int, text.strip().split()))
    return np.array(labels)


def main(feature_txt, labels_txt):
    # DATA READ
    X = parse_features(feature_txt)  # (n_samples, n_features)
    y = parse_labels(labels_txt)     # (n_samples, )
    
    # DATA SPLIT (stratified would be better, but simple split for olympiad)
    N = len(X)
    train_size = N // 2
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # TRAIN
    model = LogisticRegression(
        solver='lbfgs',
        # multi_class='multinomial',
        max_iter=2000,
        random_state=42
    )
    model.fit(X_train, y_train)

    # TEST
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    feature_txt = open("output.txt", "r").read()
    labels_txt = open("labels.txt", "r").read()
    
    accuracy = main(feature_txt, labels_txt)
    print(f"ACCURACY: {accuracy * 100:.2f}%")
