# ==============================================================
#    PROJET MMD6020 — ANALYSE DE PRÉDICTION : PIMA & DIABETES
# ==============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, brier_score_loss, ConfusionMatrixDisplay



# ============================================================
#                  FONCTIONS UTILITAIRES
# ============================================================

def metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
        "Precision": tp / (tp + fp),
        "AUC ROC": roc_auc_score(y_true, y_prob)
    }


def summarize_cv(cv):
    return {k.replace("test_", ""): np.mean(v)
            for k,v in cv.items() if k.startswith("test")}


def plot_roc(y_true, probas, label):
    fpr, tpr, _ = roc_curve(y_true, probas)
    plt.plot(fpr, tpr, label=label)


def cv_to_df(cv_results, model_name):
    return pd.DataFrame({
        "Accuracy": cv_results["test_accuracy"],
        "AUC": cv_results["test_roc_auc"],
        "Recall": cv_results["test_recall"],
        "Precision": cv_results["test_precision"],
        "Model": model_name
    })


def plot_calibration_curve(y_true, y_prob, label):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=label)


def optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    ix = np.argmax(J)
    return thresholds[ix], tpr[ix], fpr[ix]



# ============================================================
#                    GESTION DES DONNÉES
# ============================================================

def main():

# 1) CHARGEMENT DES DONNÉES
    data = pd.read_csv("proc_pima_2_withheader.csv")
    print(data.head())


# 2) REMPLACEMENT DES ZÉROS
    cols_with_zeros = ["glucose_conc", "Diastolic_BP", "Triceps_thk",
                       "2_hr_insulin", "BMI"]
    data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)
    data = data.dropna()
    
    # Convertir les étiquettes -1/1 vers 0/1
    data["Diabetes"] = ((data["Diabetes"] + 1) / 2).astype(int)


# 3) SPLIT X / y
    X = data.drop("Diabetes", axis=1)
    y = data["Diabetes"]


# 4) TRAIN / TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        )


# 5) NORMALISATION
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



# ============================================================
#          RÉGRESSION LOGISTIQUE (INFÉRENTIELLE)
# ============================================================

    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    logit_model = sm.Logit(y_train, X_train_sm)
    logit_result = logit_model.fit()

    print("\nRÉGRESSION LOGISTIQUE\n")
    print(logit_result.summary())

    params = logit_result.params
    OR_table = pd.DataFrame({
        "OR": np.exp(params),
        "CI Lower": np.exp(logit_result.conf_int()[0]),
        "CI Upper": np.exp(logit_result.conf_int()[1]),
        "p-value": logit_result.pvalues
    })
    print("\nODDS RATIOS\n", OR_table)

    y_pred_prob = logit_result.predict(X_test_sm)



# ============================================================
#          RÉGRESSION LOGISTIQUE (RIDGE/LASSO)
# ============================================================

    ridge = LogisticRegression(penalty="l2", solver="liblinear", C=1.0, max_iter=500)
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=500)

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    y_prob_ridge = ridge.predict_proba(X_test)[:, 1]
    y_prob_lasso = lasso.predict_proba(X_test)[:, 1]

    # Calibration
    ridge_cal = CalibratedClassifierCV(ridge, method="sigmoid", cv=5).fit(X_train, y_train)
    lasso_cal = CalibratedClassifierCV(lasso, method="sigmoid", cv=5).fit(X_train, y_train)

    y_prob_ridge = ridge_cal.predict_proba(X_test)[:, 1]
    y_prob_lasso = lasso_cal.predict_proba(X_test)[:, 1]

    
    
# ============================================================
#                             kNN
# ============================================================

    param_grid_knn = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 19, 21, 23, 25],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    }
    
    grid_knn = GridSearchCV(
        KNeighborsClassifier(),
        param_grid_knn,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    ).fit(X_train, y_train)
    
    print("\nkNN - MEILLEURS PARAMÈTRES :", grid_knn.best_params_)
    print("AUC (CV):", grid_knn.best_score_)

    knn = grid_knn.best_estimator_
    y_prob_knn = knn.predict_proba(X_test)[:, 1]

    # Calibration
    knn_cal = CalibratedClassifierCV(knn, method="isotonic", cv=5)
    knn_cal.fit(X_train, y_train)
    y_prob_knn = knn_cal.predict_proba(X_test)[:, 1]



# ============================================================
#              VALIDATION CROISÉE (Tous modèles)
# ============================================================

    scoring = ["accuracy", "roc_auc", "recall", "precision"]
    cv_repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    ridge_cv = cross_validate(ridge, scaler.transform(X), y, cv=cv_repeated, scoring=scoring)
    lasso_cv = cross_validate(lasso, scaler.transform(X), y, cv=cv_repeated, scoring=scoring)
    knn_cv = cross_validate(knn, scaler.transform(X), y, cv=cv_repeated, scoring=scoring)

    df_cv_all = pd.concat([
        cv_to_df(ridge_cv, "Ridge"),
        cv_to_df(lasso_cv, "LASSO"),
        cv_to_df(knn_cv, "kNN")
    ], ignore_index=True)



# ============================================================
#                  COMPARAISON DES MODÈLES
# ============================================================

    results = pd.DataFrame.from_dict({
        "Logistic": metrics(y_test, (y_pred_prob > 0.5).astype(int), y_pred_prob),
        "Ridge": metrics(y_test, ridge.predict(X_test), y_prob_ridge),
        "LASSO": metrics(y_test, lasso.predict(X_test), y_prob_lasso),
        "kNN": metrics(y_test, knn.predict(X_test), y_prob_knn)
    }, orient="index")
    print("\nPERFORMANCES COMPARÉES\n", results.round(3))

    print("\nSEUILS OPTIMAUX (Youden J)\n")
    for name, prob in {
        "Logistic": y_pred_prob,
        "Ridge": y_prob_ridge,
        "LASSO": y_prob_lasso,
        "kNN": y_prob_knn
    }.items():
        thr, sens, fpr = optimal_threshold(y_test, prob)
        spec = 1 - fpr
        print(f"{name}: seuil={thr:.3f}, sens={sens:.3f}, spéc={spec:.3f}")

    results_optimal = {}
    for name, prob in {
        "Logistic": y_pred_prob,
        "Ridge": y_prob_ridge,
        "LASSO": y_prob_lasso,
        "kNN": y_prob_knn
    }.items():
        thr, _, _ = optimal_threshold(y_test, prob)
        y_pred_opt = (prob >= thr).astype(int)
        results_optimal[name] = metrics(y_test, y_pred_opt, prob)

    print("\nPERFORMANCES AVEC SEUIL OPTIMAL\n")
    print(pd.DataFrame(results_optimal).T.round(3))




# ============================================================
#                         GRAPHIQUES
# ============================================================

    # ROC
    plt.figure()
    plot_roc(y_test, y_pred_prob, "Logistic")
    plot_roc(y_test, y_prob_ridge, "Ridge")
    plot_roc(y_test, y_prob_lasso, "LASSO")
    plot_roc(y_test, y_prob_knn, "kNN")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.title("Courbes ROC")
    plt.xlabel("1 - Spécificité")
    plt.ylabel("Sensibilité")
    plt.legend()
    plt.grid()
    plt.show()

    # Coefficents
    coef_table = pd.DataFrame({
        "Variable": X.columns,
        "Logistic": logit_result.params[1:].values,
        "Ridge": ridge.coef_[0],
        "LASSO": lasso.coef_[0]
    })
    coef_table.set_index("Variable").plot(kind="bar")
    plt.title("Coefficients comparés")
    plt.ylabel("Coefficient standardisés")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


    # Courbes de calibration
    plt.figure(figsize=(6,6))
    plot_calibration_curve(y_test, y_pred_prob, "Logistic")
    plot_calibration_curve(y_test, y_prob_ridge, "Ridge")
    plot_calibration_curve(y_test, y_prob_lasso, "LASSO")
    plot_calibration_curve(y_test, y_prob_knn, "kNN")

    plt.plot([0,1], [0,1], "--", color="gray")
    plt.title("Courbes de calibration (Reliability curves)")
    plt.xlabel("Probabilité prédite")
    plt.ylabel("Probabilité observée")
    plt.legend()
    plt.grid()
    plt.show()

    # Brier score
    print("\nBRIER SCORES")
    print("Logistic:", brier_score_loss(y_test, y_pred_prob))
    print("Ridge:", brier_score_loss(y_test, y_prob_ridge))
    print("LASSO:", brier_score_loss(y_test, y_prob_lasso))
    print("kNN:", brier_score_loss(y_test, y_prob_knn))


    # Violin plots
    for metric in ["AUC", "Accuracy", "Recall", "Precision"]:
        plt.figure(figsize=(8,6))
        sns.violinplot(data=df_cv_all, x="Model", y=metric,
                       hue="Model", palette="Set2", legend=False)
        sns.swarmplot(data=df_cv_all, x="Model", y=metric, color="black", size=5)
        plt.title(f"Distribution du score {metric} - CV 5-fold")
        plt.grid(axis="y")
        plt.show()

    
    # Matrice de confusion
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    models = {
        "Logistic": ((y_pred_prob > 0.5).astype(int)),
        "Ridge": ridge.predict(X_test),
        "LASSO": lasso.predict(X_test),
        "kNN": knn.predict(X_test)
    }
    for ax, (name, y_pred) in zip(axes.ravel(), models.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format='d'
        )
        ax.set_title(name)
    plt.suptitle("Matrices de confusion")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()