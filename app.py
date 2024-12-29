import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('insurance.csv')  # Replace with the actual dataset path

# Add custom CSS
st.markdown(
    """
    <style>
    .step-title {
        font-size: 24px;
        font-weight: bold;
        color: #2E86C1;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 18px;
        font-weight: bold;
        color: #117A65;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = "Preprocess"
if "preprocessed_data" not in st.session_state:
    st.session_state.preprocessed_data = None
if "model" not in st.session_state:
    st.session_state.model = None

# Function to navigate between pages
def navigate_to(page):
    st.session_state.page = page

# Preprocessing page
if st.session_state.page == "Preprocess":
    # Add a title and brief introduction
    st.markdown("<h1 style='text-align: center;'>Insurance Data Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align: center; font-size: 18px;'>This app allows you to preprocess insurance data and select a machine learning model for predicting insurance charges.</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='step-title'>Step 1: Preprocess the Data</div>", unsafe_allow_html=True)
    df = load_data()

    st.markdown("<div class='sub-title'>Raw Data</div>", unsafe_allow_html=True)
    st.write(df)

    st.markdown("<div class='sub-title'>Data Types</div>", unsafe_allow_html=True)
    st.write(df.dtypes)

    # Preprocessing options for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    st.markdown("<div class='sub-title'>Preprocessing Options</div>", unsafe_allow_html=True)

    preprocessing_method = {}
    for col in categorical_columns:
        with st.expander(f"Preprocess Column: {col}"):
            method = st.selectbox(f"Choose preprocessing method for '{col}'",
                                  ['None', 'Label Encoding', 'One Hot Encoding'], key=col)
            preprocessing_method[col] = method

    if st.button("Apply Preprocessing"):
        for col, method in preprocessing_method.items():
            if method == "Label Encoding":
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
            elif method == "One Hot Encoding":
                encoder = OneHotEncoder(sparse_output=False)
                transformed_data = encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(transformed_data, columns=encoder.get_feature_names_out([col]))
                df = pd.concat([df, encoded_df], axis=1).drop(columns=[col])

        st.session_state.preprocessed_data = df
        st.success("Preprocessing completed! You can proceed to model selection.")

    if st.button("Go to Model Selection"):
        navigate_to("Model Selection")

# Model selection page
elif st.session_state.page == "Model Selection":
    st.markdown("<div class='step-title'>Step 2: Select a Model</div>", unsafe_allow_html=True)

    if st.session_state.preprocessed_data is None:
        st.error("Please complete preprocessing before selecting a model.")
        st.stop()

    df = st.session_state.preprocessed_data
    st.markdown("<div class='sub-title'>Preprocessed Data</div>", unsafe_allow_html=True)
    st.write(df)

    # Model selection options
    st.markdown("<div class='sub-title'>Choose a Regression Model</div>", unsafe_allow_html=True)
    model_choice = st.selectbox("Regression Model", [
        "Linear Regression", "Ridge Regression", "Random Forest Regression",
        "Support Vector Regression (SVR)", "Decision Tree Regression", "K-Nearest Neighbors (KNN)"
    ])

    if st.button("Train Model"):
        X = df.drop(columns=["charges"])  # Assuming 'charges' is the target
        y = df["charges"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Ridge Regression":
            model = Ridge()
        elif model_choice == "Random Forest Regression":
            model = RandomForestRegressor()
        elif model_choice == "Support Vector Regression (SVR)":
            model = SVR()
        elif model_choice == "Decision Tree Regression":
            model = DecisionTreeRegressor()
        elif model_choice == "K-Nearest Neighbors (KNN)":
            model = KNeighborsRegressor()

        model.fit(X_train, y_train)
        st.session_state.model = model

        y_pred = model.predict(X_test)
        st.markdown("<div class='sub-title'>Model Performance</div>", unsafe_allow_html=True)
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R-Squared: {r2_score(y_test, y_pred):.2f}")

        st.success("Model trained successfully! You can proceed to make predictions.")

    if st.button("Go to Prediction Page"):
        navigate_to("Prediction")
    if st.button("Back to Preprocessing"):
        navigate_to("Preprocess")

# Prediction page
elif st.session_state.page == "Prediction":
    st.markdown("<div class='step-title'>Step 3: Make Predictions</div>", unsafe_allow_html=True)

    if st.session_state.model is None:
        st.error("Please train a model before making predictions.")
        st.stop()

    st.markdown("<div class='sub-title'>Enter New Data for Prediction</div>", unsafe_allow_html=True)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=5, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"])

    # Prepare input data
    new_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    st.markdown("<div class='sub-title'>Raw Input Data</div>", unsafe_allow_html=True)
    st.write(new_data)

    # Preprocess new input data
    df = load_data()
    for col in ["sex", "smoker", "region"]:
        if col in new_data:
            encoder = LabelEncoder()
            encoder.fit(df[col])
            new_data[col] = encoder.transform(new_data[col])

    st.markdown("<div class='sub-title'>Preprocessed Input Data</div>", unsafe_allow_html=True)
    st.write(new_data)

    # Predict using the trained model
    prediction = st.session_state.model.predict(new_data)
    st.write(f"Predicted Insurance Charges: ${prediction[0]:,.2f}")

    if st.button("Back to Model Selection"):
        navigate_to("Model Selection")
    if st.button("Back to Preprocessing"):
        navigate_to("Preprocess")
