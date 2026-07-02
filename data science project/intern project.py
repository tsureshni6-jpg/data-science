"""
Created on Wed Jul  1 10:11:25 2026

@author: sureshni
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Global variables
df = None
X = None
y = None
indep_selected = None
classifier = None
scaler = None

# Function to load the dataset
def load_dataset():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Info", f"Dataset Loaded Successfully!\nColumns: {', '.join(df.columns)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    else:
        messagebox.showerror("Error", "Failed to load dataset")

# Function to go to the enter variables page
def go_to_enter_variables_page():
    root.withdraw()
    enter_vars_window = tk.Toplevel(root)
    enter_vars_window.title("Enter Variables")
    enter_vars_window.geometry("800x600")
    enter_vars_window.configure(bg='#EAF4FC')  # Coral background color

    tk.Label(enter_vars_window, text="Enter dependent variable:", bg='#EAF4FC', fg='#1e3d58', font=("Helvetica", 16, 'bold')).pack(pady=10)
    dep_entry = tk.Entry(enter_vars_window, width=50, bg='#f0f8ff', font=("Arial", 12), bd=3)
    dep_entry.pack(pady=5)
    

    # Independent variables input
    # Independent Variable Label
# Independent Variable Label
    tk.Label(
    enter_vars_window,
    text="Enter independent variables:",
    bg="#EAF4FC",
    fg="#1e3d58",
    font=("Helvetica", 16, "bold")
).pack(pady=10)

# Independent Variable Entry
    indep_entry = tk.Entry(
    enter_vars_window,
    width=50,
    bg="#f0f8ff",
    font=("Arial", 12),
    bd=3
)
    indep_entry.pack(pady=5)    
    def set_variables():
        selected_dep =dep_entry.get().strip()
        global X, y, indep_selected
        try:
            # Validate and set independent and dependent variables
            if selected_dep not in df.columns:
                raise ValueError("Invalid dependent variable name.")
            
            # Automatically select all other columns as independent variables
            indep_selected = [col for col in df.columns if col != selected_dep]
            indep_entry.delete(0, tk.END)
            indep_entry.insert(0, ", ".join(indep_selected))
            
            
            # Assign X and y based on selection
            X = df[indep_selected].values
            y = df[selected_dep].values

            # Handle categorical variables by encoding them
            label_encoder = LabelEncoder()
            for col in indep_selected:
                if df[col].dtype == 'object':  # Assuming categorical columns are of type 'object'
                    df[col] = label_encoder.fit_transform(df[col])

            # Ensure all data is numeric
            X = df[indep_selected].values  # Make sure X is re-assigned after encoding

            messagebox.showinfo("Info", "Variables Set Successfully!")
            enter_vars_window.destroy()
            go_to_algorithm_page()

        except ValueError as e:
            messagebox.showerror("Error", f"Error: {e}")

    tk.Button(
        enter_vars_window,
        text="Set Variables",
        command=set_variables,
        bg="#1e3d58",
        fg="white",
        font=("Helvetica", 14, "bold"),
        relief="raised",
        bd=4
    ).pack(pady=20)


# Function to go to the algorithm selection page
def go_to_algorithm_page():
    algo_window = tk.Toplevel(root)
    algo_window.title("Select Algorithm")
    algo_window.geometry("800x600")
    algo_window.configure(bg='#EAF4FC')

    def select_algorithm():
        algo_window.destroy()
        go_to_train_model_page()

    tk.Button(algo_window, text="Select Logistic Regression", command=select_algorithm, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)

# Function to go to the training model page
def go_to_train_model_page():
    train_model_window = tk.Toplevel(root)
    train_model_window.title("Train Model")
    train_model_window.geometry("800x600")
    train_model_window.configure(bg='#EAF4FC')
    def go_back():
        train_model_window.destroy()
    go_to_algorithm_page()
    

    def train_model():
        try:
            global scaler, classifier, X_train, X_test, y_train, y_test
            # Split the dataset into training and testing sets (80-20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scaling the data (normalize features)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train the Logistic Regression model
            classifier = LogisticRegression(max_iter=3000, random_state=42)
            classifier.fit(X_train, y_train)

            # Make predictions and calculate evaluation metrics
            y_pred = classifier.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)

            result_text = (
                f"Model Trained Successfully!\n\n"
                f"Accuracy: {score * 100:.2f}%\n"
                f"Confusion Matrix:\n{conf_matrix}\n\n"
                f"Classification Report:\n{class_report}"
            )
            messagebox.showinfo("Model Result", result_text)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while training the model: {e}")

    tk.Button(train_model_window, text="Train Model", command=train_model, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)
    tk.Button(
    train_model_window,
    text="Back",
    command=go_back,
    bg="#1e3d58",
    fg="white",
    font=("Helvetica", 14, "bold")
).pack(side=tk.LEFT, padx=80, pady=20)

    tk.Button(
    train_model_window,
    text="Next",
    command=go_to_prediction_page,
    bg="#1e3d58",
    fg="white",
    font=("Helvetica", 14, "bold")
).pack(side=tk.RIGHT, padx=80, pady=20)
    tk.Button(train_model_window, text="Go to Prediction Page", command=go_to_prediction_page, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)

# Function to go to the prediction page
def go_to_prediction_page():
    prediction_window = tk.Toplevel(root)
    prediction_window.title("Enter Input Values")
    prediction_window.geometry("800x600")
    prediction_window.configure(bg='#EAF4FC')

    inputs = []

    for feature in indep_selected:
        frame = tk.Frame(prediction_window, bg='#EAF4FC')
        frame.pack(pady=5,anchor="center")
        label = tk.Label(
    frame,
    text=f"Enter value for {feature}:",
    width=28,
    anchor="w",
    bg="#EAF4FC",
    fg="#1e3d58",
    font=("Helvetica", 14)
)
        label.pack(side=tk.LEFT, padx=(80,10))
        entry = tk.Entry(frame, bg='#f0f8ff', font=("Arial", 12), bd=3)
        entry.pack(side=tk.RIGHT, padx=(10,20))
        inputs.append(entry)

    def predict():
        try:
            input_values = [float(entry.get()) for entry in inputs]  # Get input values from the user
            input_array = np.array(input_values).reshape(1, -1)  # Reshape the input values
            input_array = scaler.transform(input_array)  # Scale input data using the fitted scaler
            
            prediction = classifier.predict(input_array)  # Get the prediction from the classifier
            
            # Display a message depending on the prediction result
            if prediction[0] == 1:
                messagebox.showinfo("Prediction", "Predicted: Diabetes")
            else:
                messagebox.showinfo("Prediction", "Predicted: No Diabetes")
    
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")

    tk.Button(prediction_window, text="Predict", command=predict, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)

# Main window
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("800x600")
root.configure(bg='#EAF4FC')
# Title
title_label = tk.Label(
    root,
    text="DIABETES PREDICTION SYSTEM",
    bg="#EAF4FC",
    fg="#1e3d58",
    font=("Helvetica", 24, "bold")
)
title_label.pack(pady=30)

tk.Button(root, text="Load Dataset", command=load_dataset, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)
tk.Button(root, text="Next", command=go_to_enter_variables_page, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)

root.mainloop()
