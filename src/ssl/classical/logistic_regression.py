from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
  
def train_logistic_regression(features, labels):

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=5000, multi_class="multinomial", solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # return everything needed for confusion matrix
    return clf, scaler, (X_test, y_test, y_pred)


def plot_confusion_matrix(results, labels, class_names=None, out_prefix="logreg"):
    """
    Plot and save confusion matrix (counts + normalized).

    Parameters
    ----------
    results : tuple from train_logistic_regression (X_test, y_test, y_pred)
    labels : array-like of all labels (to get consistent ordering)
    class_names : list of str or None
    out_prefix : str prefix for saved PNGs
    """

    X_test, y_test, y_pred = results

    labels_order = np.unique(labels)
    lbls = class_names if class_names is not None else labels_order

    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Counts
    plt.figure(figsize=(6.5, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    ticks = np.arange(len(cm))
    plt.xticks(ticks, lbls, rotation=45, ha="right"); plt.yticks(ticks, lbls)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confusion_counts.png", dpi=200)
    plt.close()

    # Normalized
    plt.figure(figsize=(6.5, 6))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title("Normalized Confusion")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.xticks(ticks, lbls, rotation=45, ha="right"); plt.yticks(ticks, lbls)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confusion_normalized.png", dpi=200)
    plt.close()


def label_efficiency_curve(
    features,
    labels,
    *,
    fractions=(0.01, 0.05, 0.10, 0.25, 0.50, 1.00),
    seeds=(0, 1, 2, 3, 4),
    test_size=0.20,
    random_state=42,
    title="Label-efficiency",
    out_file="label_efficiency.png",
    max_iter=5000,
    verbose=True,
):
    """
    Compute macro-F1 vs labeled fraction using a multinomial logistic-regression probe.
    Uses a fixed stratified train/test split for comparability, then subsamples
    the training set with stratification at each fraction.

    Returns
    -------
    xs : np.ndarray   (percent labeled, e.g., [1, 5, 10, ...])
    ys : np.ndarray   (mean macro-F1 across seeds)
    yerr : np.ndarray (std of macro-F1 across seeds)
    """

    # Fixed split to mirror your training/eval routine
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size, random_state=random_state,
        stratify=labels
    )

    C = len(np.unique(y_train))  # number of classes
    N = len(y_train)             # training size

    results = []  # (fraction_req, mean_f1, std_f1, eff_frac_used)

    for f in fractions:
        f1s = []
        eff_fracs = []
        for s in seeds:
            # Build a stratified subset of the training set.
            if f == 1.0:
                sub_idx = np.arange(N)
            else:
                n_train = int(np.ceil(f * N))
                n_train = max(n_train, C)     # >=1 per class in subset
                n_train = min(n_train, N - C) # leave >=1 per class in remainder
                if n_train < C:
                    # Not feasible at this size — skip this (fraction, seed) combo
                    continue
                sss = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=s)
                (sub_idx, _), = sss.split(X_train, y_train)

            # Fit scaler on the current subset only; transform the fixed test set
            scaler = StandardScaler()
            X_tr_sub = scaler.fit_transform(X_train[sub_idx])
            X_te_sub = scaler.transform(X_test)

            # Same probe as elsewhere
            clf = LogisticRegression(max_iter=max_iter, multi_class="multinomial", solver="lbfgs")
            clf.fit(X_tr_sub, y_train[sub_idx])

            y_hat = clf.predict(X_te_sub)
            f1m = f1_score(y_test, y_hat, average="macro")
            f1s.append(f1m)
            eff_fracs.append(len(sub_idx) / N)

        if f1s:
            mean_f1 = float(np.mean(f1s))
            std_f1  = float(np.std(f1s, ddof=0))
            eff_frac = float(np.mean(eff_fracs))
            results.append((f, mean_f1, std_f1, eff_frac))
            if verbose:
                print(f"{f*100:5.1f}% labels  ->  macro-F1 = {mean_f1:.3f} ± {std_f1:.3f} "
                      f"(effective {eff_frac*100:.1f}% of train)")

    if not results:
        if verbose:
            print("No feasible fractions produced results (check class counts vs tiny fractions).")
        return np.array([]), np.array([]), np.array([])

    xs   = np.array([r[0]*100 for r in results])
    ys   = np.array([r[1]     for r in results])
    yerr = np.array([r[2]     for r in results])

    # Plot
    plt.figure()
    plt.errorbar(xs, ys, yerr=yerr, marker="o")
    plt.xlabel("Labeled data used (%)")
    plt.ylabel("Macro-F1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()

    return xs, ys, yerr


