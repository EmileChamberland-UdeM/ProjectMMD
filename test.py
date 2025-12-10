# ===============================================================
#    PROJET MMD6020 — ANALYSE DE PRÉDICTION : PIMA & DIABETES
#    Version entièrement réécrite (sans leakage, CV correcte)
# ===============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_validate, GridSearchCV, StratifiedKFold,
    RepeatedStratifiedKFold
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score, brier_score_loss,
    precision_recall_curve, average_precision_score
)

# ===============================================================
#    FONCTIONS UTILITAIRES
# ===============================================================

def metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
        "Precision": tp / (tp + fp) if (tp+fp)>0 else np.nan,
        "AUC ROC": roc_auc_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob)
    }


def plot_roc(y_true, prob, label):
    fpr, tpr, _ = roc_curve(y_true, prob)
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, prob):.3f})")


def plot_pr(y_true, prob, label):
    prec, rec, _ = precision_recall_curve(y_true, prob)
    auprc = average_precision_score(y_true, prob)
    plt.plot(rec, prec, label=f"{label} (AUPRC={auprc:.3f})")


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE approximé."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        acc_bin = np.mean(y_true[mask])
        conf_bin = np.mean(y_prob[mask])
        ece += np.abs(acc_bin - conf_bin) * (mask.sum() / len(y_true))
    return ece


def optimal_threshold_cv(model, X, y):
    """Seuil optimal déterminé par Youden J via CV interne."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, thr = roc_curve(y_val, prob)
        J = tpr - fpr
        thresholds.append(thr[np.argmax(J)])

    return np.mean(thresholds)


def compute_vif(df):
    """Variance Inflation Factors (multicolinéarité)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_vif = sm.add_constant(df)
    return pd.DataFrame({
        "Variable": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    })


def confusion_mats(y_true, prob, thr, name):
    pred_05  = (prob >= 0.5).astype(int)
    pred_opt = (prob >= thr).astype(int)

    print(f"\n--- {name} : Matrice de confusion (seuil 0.5) ---")
    print(confusion_matrix(y_true, pred_05))

    print(f"\n--- {name} : Matrice de confusion (seuil optimal {thr:.3f}) ---")
    print(confusion_matrix(y_true, pred_opt))


# ===============================================================
#    MAIN PIPELINE
# ===============================================================

def main():

    # -----------------------------------------------------------
    # 1. DONNÉES
    # -----------------------------------------------------------
    data = pd.read_csv("proc_pima_2_withheader.csv")

    cols_with_zeros = ["glucose_conc", "Diastolic_BP", "Triceps_thk",
                       "2_hr_insulin", "BMI"]
    data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)

    # Split X / y
    X = data.drop("Diabetes", axis=1)
    y = data["Diabetes"]

    # Labels doivent être 0/1
    y = ((y + 1) / 2).astype(int)

    # -----------------------------------------------------------
    # 2. VIF — Évaluation initiale de la multicolinéarité
    # -----------------------------------------------------------
    print("\n=== VIF (multicolinéarité) ===")
    print(compute_vif(X))

    # -----------------------------------------------------------
    # 3. SPLIT TRAIN / TEST
    # -----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -----------------------------------------------------------
    # 4. PIPELINES DES MODÈLES (sans fuite)
    # -----------------------------------------------------------
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=500))
    ])

    lasso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=500))
    ])

    knn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])

    # -----------------------------------------------------------
    # 5. GRID SEARCH POUR kNN
    # -----------------------------------------------------------
    param_grid_knn = {
        "clf__n_neighbors": [3,5,7,9,11,15,19],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"]
    }

    grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring="roc_auc", n_jobs=-1)
    grid_knn.fit(X_train, y_train)
    knn = grid_knn.best_estimator_

    print("\nkNN — Meilleurs paramètres :", grid_knn.best_params_)

    # -----------------------------------------------------------
    # 6. CV interne : seuil optimal (sans toucher au test)
    # -----------------------------------------------------------
    thr_ridge = optimal_threshold_cv(ridge, X_train, y_train)
    thr_lasso = optimal_threshold_cv(lasso, X_train, y_train)
    thr_knn   = optimal_threshold_cv(knn,   X_train, y_train)

    # -----------------------------------------------------------
    # 7. RÉGRESSION LOGISTIQUE INFÉRENTIELLE (statsmodels)
    # -----------------------------------------------------------
    # Imputation/standardisation manuelle pour statsmodels
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    X_train_sm = sm.add_constant(X_train_scaled)
    X_test_sm  = sm.add_constant(X_test_scaled)

    logit = sm.Logit(y_train, X_train_sm).fit()
    print("\n=== LOGIT (inférentiel) ===")
    print(logit.summary())

    # Seuil logit via CV
    prob_logit_train = logit.predict(X_train_sm)
    # CV custom
    def logit_thr_cv():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        thr_list = []
        for tr, val in skf.split(X_train_scaled, y_train):
            m = sm.Logit(y_train.iloc[tr], sm.add_constant(X_train_scaled[tr])).fit(disp=0)
            p = m.predict(sm.add_constant(X_train_scaled[val]))
            fpr, tpr, thr = roc_curve(y_train.iloc[val], p)
            J = tpr - fpr
            thr_list.append(thr[np.argmax(J)])
        return np.mean(thr_list)

    thr_logit = logit_thr_cv()

    # -----------------------------------------------------------
    # 8. PRÉDICTION SUR LE TEST
    # -----------------------------------------------------------
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    y_prob_ridge = ridge.predict_proba(X_test)[:,1]
    y_prob_lasso = lasso.predict_proba(X_test)[:,1]
    y_prob_knn   = knn.predict_proba(X_test)[:,1]
    y_prob_logit = logit.predict(X_test_sm)

    # -----------------------------------------------------------
    # 9. TABLEAU DES PERFORMANCES
    # -----------------------------------------------------------
    results = pd.DataFrame({
        "Logit": metrics(y_test, (y_prob_logit>=0.5).astype(int), y_prob_logit),
        "Ridge": metrics(y_test, ridge.predict(X_test), y_prob_ridge),
        "LASSO": metrics(y_test, lasso.predict(X_test), y_prob_lasso),
        "kNN":   metrics(y_test, knn.predict(X_test),   y_prob_knn)
    }).T

    print("\n=== PERFORMANCES BRUTES — SEUIL 0.5 ===")
    print(results.round(3))

    # -----------------------------------------------------------
    # 10. MATRICES DE CONFUSION avec seuil optimal
    # -----------------------------------------------------------
    print("\n=== MATRICES DE CONFUSION (seuil optimal) ===")
    confusion_mats(y_test, y_prob_logit, thr_logit, "Logit")
    confusion_mats(y_test, y_prob_ridge, thr_ridge, "Ridge")
    confusion_mats(y_test, y_prob_lasso, thr_lasso, "LASSO")
    confusion_mats(y_test, y_prob_knn,   thr_knn,   "kNN")

    # -----------------------------------------------------------
    # 11. CALIBRATION — courbes + ECE
    # -----------------------------------------------------------
    print("\n=== ECE (Expected Calibration Error) ===")
    print("Logit :", expected_calibration_error(y_test, y_prob_logit))
    print("Ridge :", expected_calibration_error(y_test, y_prob_ridge))
    print("LASSO :", expected_calibration_error(y_test, y_prob_lasso))
    print("kNN   :", expected_calibration_error(y_test, y_prob_knn))

    # Courbes calibration
    from sklearn.calibration import calibration_curve
    plt.figure()
    for prob, label in [(y_prob_logit,"Logit"),(y_prob_ridge,"Ridge"),
                        (y_prob_lasso,"LASSO"),(y_prob_knn,"kNN")]:
        true, pred = calibration_curve(y_test, prob, n_bins=10)
        plt.plot(pred, true, marker='o', label=label)
    plt.plot([0,1],[0,1],'--')
    plt.title("Courbes de calibration")
    plt.xlabel("Probabilité prédite")
    plt.ylabel("Probabilité observée")
    plt.legend()
    plt.grid()
    plt.show()

    # -----------------------------------------------------------
    # 12. ROC
    # -----------------------------------------------------------
    plt.figure()
    plot_roc(y_test, y_prob_logit, "Logit")
    plot_roc(y_test, y_prob_ridge, "Ridge")
    plot_roc(y_test, y_prob_lasso, "LASSO")
    plot_roc(y_test, y_prob_knn,   "kNN")
    plt.plot([0,1],[0,1],"--")
    plt.title("ROC curves")
    plt.xlabel("1 - Spécificité (FPR)")
    plt.ylabel("Sensibilité (TPR)")
    plt.legend()
    plt.grid()
    plt.show()

    # -----------------------------------------------------------
    # 13. Precision–Recall
    # -----------------------------------------------------------
    plt.figure()
    plot_pr(y_test, y_prob_logit, "Logit")
    plot_pr(y_test, y_prob_ridge, "Ridge")
    plot_pr(y_test, y_prob_lasso, "LASSO")
    plot_pr(y_test, y_prob_knn,   "kNN")
    plt.title("Precision–Recall curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.show()


# ===============================================================
#    RUN
# ===============================================================

if __name__ == "__main__":
    main()
