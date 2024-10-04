#!./env/bin/python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from perceptron.eda import EDA
from perceptron.perceptron import Perceptron
from perceptron.utils import get_metrics


def load_data() -> pd.DataFrame:
    return pd.read_parquet("./diabetes.parquet")


if __name__ == "__main__":
    random_state = 112
    df = load_data()
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
    # eda.get_histogram(x_axis_titles)

    df_train, _df = train_test_split(df, test_size=0.3, random_state=random_state)
    df_validate, df_test = train_test_split(_df, test_size=0.5, random_state=random_state)

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

    print("perceptron")
    perceptron = Perceptron()
    perceptron.fit(
        x_train,
        y_train,
        learning_rate=0.01,
        epochs=100,
        random_seed=random_state,
        early_stopping=20,
        x_validate=x_validate,
        y_validate=y_validate,
    )

    predictions = perceptron.predict(x_validate)
    df_validate = df_validate.assign(prediction=predictions)
    print("validation")
    get_metrics(df_validate.Outcome, df_validate.prediction)

    predictions = perceptron.predict(x_test)
    df_test = df_test.assign(prediction=predictions)
    print("test")
    get_metrics(df_test.Outcome, df_test.prediction)
