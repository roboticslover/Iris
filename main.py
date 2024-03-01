import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Function to standardize user input and make predictions


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    scaled_input = scaler.transform(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(scaled_input)
    species = ['Setosa', 'Versicolor', 'Virginica']
    return species[prediction[0]]


# Streamlit app layout
st.title("Iris Species Prediction")

# User input sliders with descriptive labels and informative hover messages
st.subheader("Enter flower measurements:")
sepal_length = st.slider(
    "Sepal Length (mm)",
    float(X[:, 0].min()),
    float(X[:, 0].max()),
    float(X[:, 0].mean()),
    key="sepal_length",
    help="Length of the sepal in millimeters",
)
sepal_width = st.slider(
    "Sepal Width (mm)",
    float(X[:, 1].min()),
    float(X[:, 1].max()),
    float(X[:, 1].mean()),
    key="sepal_width",
    help="Width of the sepal in millimeters",
)
petal_length = st.slider(
    "Petal Length (mm)",
    float(X[:, 2].min()),
    float(X[:, 2].max()),
    float(X[:, 2].mean()),
    key="petal_length",
    help="Length of the petal in millimeters",
)
petal_width = st.slider(
    "Petal Width (mm)",
    float(X[:, 3].min()),
    float(X[:, 3].max()),
    float(X[:, 3].mean()),
    key="petal_width",
    help="Width of the petal in millimeters",
)

# Run model button with a clear and concise label
if st.button("Predict Species"):
    # Make the prediction
    predicted_species = predict_species(
        sepal_length, sepal_width, petal_length, petal_width)

    # Display the prediction with clear formatting and informative text
    st.header("Prediction")
    st.success(f"The predicted Iris species is: {predicted_species}")
    st.write("**Note:** This prediction is based on a trained machine learning model and may not be perfectly accurate.")
