
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error


def main():
    data = pd.read_csv("proc_pima_2_withheader.csv")

    X = data.drop("Diabetes", axis=1)
    y = data["Diabetes"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)

    lasso_cv_model = LassoCV(cv=5, random_state=42)
    lasso_cv_model.fit(X_train, y_train)

    # y_pred = lasso_model.predict(X_test)
    y_pred = lasso_cv_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # coefficients = lasso_model.coef_
    coefficients = lasso_cv_model.coef_
    feature_coefficients = pd.DataFrame({"Feature": X.columns, "Coefficients": coefficients})
    print(feature_coefficients)

    selected_features = feature_coefficients[feature_coefficients['Coefficients'] !=0]
    print("\nSelected Features:")
    print(selected_features)





if __name__ == "__main__":
    main()




# print(data[data["times_pregnant"] > 8])

#    # Challenge : Imprimer les plus jeunes personnes pour chaque nombre de grosesses
#    results = []
#    list = sorted(data["times_pregnant"].unique())
#
#    for items in list:
#        hold = 1
#        for i, row in data.iterrows():
#            if row["times_pregnant"] == items:
#                if row["Age"] > hold:
#                    hold = row["Age"]
#        results.append(hold)
#
#    print(results)