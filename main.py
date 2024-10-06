#!./env/bin/python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from perceptron.eda import EDA
from perceptron.perceptron import Perceptron
from perceptron.utils import get_metrics

if __name__ == "__main__":
    random_state = 112
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    df = pd.read_parquet("./diabetes.parquet")
    eda = EDA(df)

    x_axis_titles = {
        "Pregnancies": "# of times pregnant",
        "Glucose": "Plasma glucose concentration",
        "BloodPressure": "Blood pressure (mm Hg)",
        "SkinThickness": "Triceps skin fold thickness (mm)",
        "Insulin": "Insulin (µU/ml)",
        "BMI": "BMI (kg/m^2)",
        "DiabetesPedigreeFunction": "Diabetes pedigree function",
        "Age": "Age (years)",
        "Outcome": "Outcome",
    }

    eda.show_missing()
    # df.Insulin = np.log(df.Insulin + 20)
    eda.get_histogram(x_axis_titles)

    # split dataset
    df_train, _df = train_test_split(df, test_size=0.3, random_state=random_state, stratify=df.Outcome)
    df_validate, df_test = train_test_split(_df, test_size=0.5, random_state=random_state, stratify=_df.Outcome)

    x_train = df_train.drop(columns=["Outcome"])
    y_train = df_train.Outcome
    x_validate = df_validate.drop(columns=["Outcome"])
    y_validate = df_validate.Outcome
    x_test = df_test.drop(columns=["Outcome"])
    y_test = df_test.Outcome

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_validate = scaler.transform(x_validate)
    x_test = scaler.transform(x_test)

    print(f"train set: {x_train.shape}")
    print(f"validation set: {x_validate.shape}")
    print(f"test set: {x_test.shape}")

    # perceptron
    print("perceptron")
    best_acc = 0.0
    best_learning_rate = 0.0
    for learning_rate in learning_rates:
        perceptron = Perceptron()
        perceptron.fit(
            x_train,
            y_train,
            learning_rate=learning_rate,
            epochs=10000,
            random_seed=random_state,
            early_stopping=20,
            x_validate=x_validate,
            y_validate=y_validate,
        )
        if perceptron.best_acc > best_acc:
            best_acc = perceptron.best_acc
            best_learning_rate = learning_rate

    print(f"best learning rate: {best_learning_rate:.4f}")
    print(f"best acc: {best_acc:.4f}")
    perceptron = Perceptron()
    perceptron.fit(
        x_train,
        y_train,
        learning_rate=best_learning_rate,
        epochs=10000,
        random_seed=random_state,
        early_stopping=20,
        x_validate=x_validate,
        y_validate=y_validate,
    )

    # logistic regression
    print("logistic regression")
    log_reg = LogisticRegression(random_state=random_state)
    log_reg.fit(x_train, y_train)

    models = {
        "perceptron": perceptron,
        "logistic regression": log_reg,
    }
    datasets = {
        "train": [x_train, y_train],
        "validate": [x_validate, y_validate],
        "test": [x_test, y_test],
    }
    for model_name, model in models.items():
        print(model_name)
        for dataset_name, (x, y) in datasets.items():
            print(dataset_name)
            predictions = model.predict(x)
            get_metrics(y, predictions)

    # with imputation
    print("imputation")
    impute_values = {
        "Glucose": df_train.Glucose.mean(),
        "BloodPressure": df_train.BloodPressure.mean(),
        "SkinThickness": df_train.SkinThickness.mean(),
        "BMI": df_train.BMI.mean(),
    }
    for col, value in impute_values.items():
        df_train[col] = df_train[col].replace(0, value)
        df_validate[col] = df_validate[col].replace(0, value)
        df_test[col] = df_test[col].replace(0, value)

    x_train = df_train.drop(columns=["Outcome"])
    x_validate = df_validate.drop(columns=["Outcome"])
    x_test = df_test.drop(columns=["Outcome"])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_validate = scaler.transform(x_validate)
    x_test = scaler.transform(x_test)

    # perceptron
    print("perceptron")
    best_acc = 0.0
    best_learning_rate = 0.0
    for learning_rate in learning_rates:
        perceptron = Perceptron()
        perceptron.fit(
            x_train,
            y_train,
            learning_rate=learning_rate,
            epochs=10000,
            random_seed=random_state,
            early_stopping=20,
            x_validate=x_validate,
            y_validate=y_validate,
        )
        if perceptron.best_acc > best_acc:
            best_acc = perceptron.best_acc
            best_learning_rate = learning_rate

    print(f"best learning rate: {best_learning_rate:.4f}")
    print(f"best acc: {best_acc:.4f}")
    perceptron = Perceptron()
    perceptron.fit(
        x_train,
        y_train,
        learning_rate=best_learning_rate,
        epochs=10000,
        random_seed=random_state,
        early_stopping=20,
        x_validate=x_validate,
        y_validate=y_validate,
    )

    # logistic regression
    print("logistic regression")
    log_reg = LogisticRegression(random_state=random_state)
    log_reg.fit(x_train, y_train)

    models = {
        "perceptron": perceptron,
        "logistic regression": log_reg,
    }
    datasets = {
        "train": [x_train, y_train],
        "validate": [x_validate, y_validate],
        "test": [x_test, y_test],
    }
    for model_name, model in models.items():
        print(model_name)
        for dataset_name, (x, y) in datasets.items():
            print(dataset_name)
            predictions = model.predict(x)
            get_metrics(y, predictions)
