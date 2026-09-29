import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Page title
# -----------------------------

st.set_page_config(
    page_title="KNN Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("K-Nearest Neighbors Classifier")
st.write(
    "This app demonstrates how the K-nearest-neighbors algorithm "
    "can be used to classify heart-disease observations."
)

# -----------------------------
# Load and prepare the data
# -----------------------------

@st.cache_data
def load_data():
    data = pd.read_excel("dataset.xls")

    # Remove the row-number column
    if "row" in data.columns:
        data = data.drop(columns=["row"])

    # Convert categorical variables to numerical variables
    data = pd.get_dummies(
        data,
        columns=["famhist"],
        drop_first=True,
        dtype=float
    )

    return data


df = load_data()

X = df.drop(columns=["chd"])
y = df["chd"]

# Split the data before standardization
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=101,
    stratify=y
)

# -----------------------------
# App tabs
# -----------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["Explore Data", "Choose K", "Evaluate Model", "Make a Prediction"]
)

# -----------------------------
# Tab 1: Explore the data
# -----------------------------

with tab1:
    st.header("Explore the Data")

    st.write("The first five observations are:")

    # Display original-style data
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Number of observations", df.shape[0])

    with col2:
        st.metric("Number of predictors", X.shape[1])

    with col3:
        st.metric("Number of classes", y.nunique())

    st.subheader("Class distribution")

    class_counts = y.value_counts().sort_index()
    class_counts.index = ["No heart disease", "Heart disease"]

    st.bar_chart(class_counts)

    st.write(
        """
        The target variable `chd` is coded as follows:

        - `0`: No coronary heart disease
        - `1`: Coronary heart disease
        """
    )

# -----------------------------
# Tab 2: Choose K
# -----------------------------

with tab2:
    st.header("Choose the Number of Neighbors")

    st.write(
        """
        The value of K determines how many nearby observations are used
        to classify a new observation. Cross-validation is used to compare
        different values of K using the training data.
        """
    )

    k_values = range(1, 40)
    validation_errors = []

    for k_value in k_values:
        validation_model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k_value))
        ])

        scores = cross_val_score(
            validation_model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        validation_errors.append(1 - scores.mean())

    best_k = list(k_values)[validation_errors.index(min(validation_errors))]

    st.success(
        f"The value of K with the smallest cross-validation error is {best_k}."
    )

    fig, ax = plt.subplots()

    ax.plot(
        list(k_values),
        validation_errors,
        marker="o",
        linestyle="--"
    )

    ax.set_xlabel("Number of neighbors, K")
    ax.set_ylabel("Cross-validation error rate")
    ax.set_title("Cross-validation Error Rate versus K")

    st.pyplot(fig)

    st.write(
        """
        Smaller values of K can produce a flexible model that is sensitive
        to individual observations. Larger values of K produce a smoother
        decision rule.
        """
    )

# -----------------------------
# Tab 3: Evaluate the model
# -----------------------------

with tab3:
    st.header("Evaluate the KNN Model")

    selected_k = st.slider(
        "Select a value of K",
        min_value=1,
        max_value=39,
        value=best_k
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=selected_k))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    error_rate = 1 - accuracy

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Test accuracy", f"{accuracy:.2%}")

    with col2:
        st.metric("Test error rate", f"{error_rate:.2%}")

    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(y_test, predictions)

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual: No CHD", "Actual: CHD"],
        columns=["Predicted: No CHD", "Predicted: CHD"]
    )

    st.dataframe(matrix_df)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=["No CHD", "CHD"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3))

# -----------------------------
# Tab 4: Make a prediction
# -----------------------------

with tab4:
    st.header("Predict for a New Patient")

    st.write(
        """
        Enter values for a new observation. The prediction is for educational
        purposes only and is not a medical diagnosis.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        sbp = st.number_input(
            "Systolic blood pressure",
            min_value=50.0,
            max_value=250.0,
            value=130.0
        )

        tobacco = st.number_input(
            "Tobacco consumption",
            min_value=0.0,
            max_value=50.0,
            value=3.0
        )

        ldl = st.number_input(
            "LDL cholesterol",
            min_value=0.0,
            max_value=20.0,
            value=4.0
        )

        adiposity = st.number_input(
            "Adiposity",
            min_value=0.0,
            max_value=50.0,
            value=25.0
        )

        famhist = st.selectbox(
            "Family history of heart disease",
            ["Absent", "Present"]
        )

    with col2:
        typea = st.number_input(
            "Type-A behavior score",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

        obesity = st.number_input(
            "Obesity",
            min_value=0.0,
            max_value=50.0,
            value=25.0
        )

        alcohol = st.number_input(
            "Alcohol consumption",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )

        age = st.number_input(
            "Age",
            min_value=1.0,
            max_value=120.0,
            value=50.0
        )

    if st.button("Predict"):
        new_patient = pd.DataFrame({
            "sbp": [sbp],
            "tobacco": [tobacco],
            "ldl": [ldl],
            "adiposity": [adiposity],
            "typea": [typea],
            "obesity": [obesity],
            "alcohol": [alcohol],
            "age": [age],
            "famhist_Present": [
                1.0 if famhist == "Present" else 0.0
            ]
        })

        # Ensure the new observation has exactly the same columns as X
        new_patient = new_patient.reindex(
            columns=X.columns,
            fill_value=0
        )

        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=best_k))
        ])

        final_model.fit(X_train, y_train)

        prediction = final_model.predict(new_patient)[0]
        probabilities = final_model.predict_proba(new_patient)[0]

        if prediction == 1:
            st.warning(
                "The model predicts the presence of coronary heart disease."
            )
        else:
            st.success(
                "The model predicts the absence of coronary heart disease."
            )

        probability_df = pd.DataFrame({
            "Class": ["No CHD", "CHD"],
            "Predicted probability": probabilities
        })

        st.dataframe(probability_df)