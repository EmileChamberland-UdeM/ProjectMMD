import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score
)
import statsmodels.api as sm

def main():
    # Importation des données
    data = pd.read_csv("proc_pima_2_withheader.csv")

    # Séparation X / y
    X = data.drop("Diabetes", axis=1)
    y = data["Diabetes"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Régression logistique avec pénalisation L1 (équivalent LASSO)
    logit_cv = LogisticRegressionCV(
        Cs=10,            # nombre de valeurs de régularisation testées
        cv=5,             # nombre de folds pour la CV
        penalty="l1",     # pénalisation LASSO
        solver="saga",    # solver compatible avec L1
        max_iter=5000
    )
    logit_cv.fit(X_train, y_train)

    # Prédictions
    y_pred = logit_cv.predict(X_test)
    y_prob = logit_cv.predict_proba(X_test)[:, 1]

    # Performances prédictives
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Précision:", precision_score(y_test, y_pred))
    print("Sensibilité (Recall):", recall_score(y_test, y_pred))
    print("Spécificité:", confusion_matrix(y_test, y_pred)[0,0] / sum(y_test==0))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))

    # Statistiques inférentielles (via statsmodels)
    logit_sm = sm.Logit(y_train, sm.add_constant(X_train))
    result = logit_sm.fit()
    print(result.summary())  # coefficients, significativité (p-values), tailles d’effet (odds ratios)

    # Odds Ratios (tailles d’effet)
    odds_ratios = pd.DataFrame({
        "Variable": X_train.columns,
        "OR": result.params[1:].apply(lambda x: round(np.exp(x), 3)),
        "p-value": result.pvalues[1:]
    })
    print("\nTailles d’effet (Odds Ratios):\n", odds_ratios)

if __name__ == "__main__":
    import numpy as np
    main()
