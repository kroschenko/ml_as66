import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

def regression_part(happiness_path):
    df = pd.read_csv(happiness_path)
    X = df[["GDP per capita"]]
    y = df["Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    y_train_pred = lin_reg.predict(X_train)
    y_test_pred = lin_reg.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print("=== РЕГРЕССИЯ (World Happiness Report) ===")
    print(f"Train -> MSE: {mse_train:.3f}, R²: {r2_train:.3f}")
    print(f"Test  -> MSE: {mse_test:.3f}, R²: {r2_test:.3f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Train
    axes[0].scatter(X_train, y_train, alpha=0.7, label="Train data")
    axes[0].plot(X_train, y_train_pred, color="red", linewidth=2, label="Regression line")
    axes[0].set_title("Train data")
    axes[0].set_xlabel("GDP per capita")
    axes[0].set_ylabel("Happiness Score")
    axes[0].legend()
    axes[0].grid(True)

    # Test
    axes[1].scatter(X_test, y_test, alpha=0.7, label="Test data")
    axes[1].plot(X_test, y_test_pred, color="red", linewidth=2, label="Regression line")
    axes[1].set_title("Test data")
    axes[1].set_xlabel("GDP per capita")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("Линейная регрессия: Train vs Test")
    plt.tight_layout()
    plt.show()

def classification_part(churn_path):

    df = pd.read_csv(churn_path)
    df = pd.get_dummies(df.drop(["customerID"], axis=1), drop_first=True)
    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)


    y_train_pred = log_reg.predict(X_train)
    y_test_pred = log_reg.predict(X_test)


    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)

    print("=== КЛАССИФИКАЦИЯ (Telco Customer Churn) ===")
    print(f"Test -> Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}\n")


    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axes[0])
    axes[0].set_title("Confusion Matrix (Train)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axes[1])
    axes[1].set_title("Confusion Matrix (Test)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.suptitle("Confusion Matrices: Train vs Test")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    happiness_path = "world_happiness_report.csv"  
    churn_path = "Telco-Customer-Churn.csv"      

    regression_part(happiness_path)
    classification_part(churn_path)
