import joblib
import pandas as pd

# Load saved model and encoder
model = joblib.load("models/student_model.pkl")
encoder = joblib.load("models/encoder.pkl")

print("Model loaded successfully!")
print("Encoder loaded successfully!")


# Prediction function
def predict_student_score(features):

    prediction = model.predict(features)

    print("\n===== Prediction =====")
    print("Predicted Math Score:", round(prediction[0], 2))


while True:

    # User Input
    gender = input("Enter Gender (male/female): ")

    race = input(
        "Enter Race (group A/group B/group C/group D/group E): "
    )

    parent_education = input(
        "Enter Parent Education: "
    )

    lunch = input(
        "Enter Lunch (standard/(free/reduced)): "
    )

    test_prep = input(
        "Enter Test Preparation (completed/none): "
    )

    reading_score = int(
        input("Enter Reading Score: ")
    )

    writing_score = int(
        input("Enter Writing Score: ")
    )

    # Create student dictionary
    student = {
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent_education,
        "lunch": lunch,
        "test preparation course": test_prep,
        "reading score": reading_score,
        "writing score": writing_score
    }

    # Convert dictionary to DataFrame
    student_df = pd.DataFrame([student])

    print("\n===== Student Data =====")
    print(student_df)

    # Categorical columns
    categorical_columns = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course"
    ]

    # Encode categorical data
    encoded_student = encoder.transform(
        student_df[categorical_columns]
    )

    encoded_student_df = pd.DataFrame(
        encoded_student,
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )

    print("\n===== Encoded Student =====")
    print(encoded_student_df)

    # Numerical columns
    numerical_student = student_df[
        ["reading score", "writing score"]
    ]

    print("\n===== Numerical Student Data =====")
    print(numerical_student)

    # Final feature matrix
    final_student = pd.concat(
        [encoded_student_df, numerical_student],
        axis=1
    )

    print("\n===== Final Student Features =====")
    print(final_student)

    print(final_student.shape)

    # Predict score
    predict_student_score(final_student)

    # Continue?
    again = input(
        "\nPredict another student? (y/n): "
    )

    if again.lower() != "y":
        print("Goodbye!")
        break