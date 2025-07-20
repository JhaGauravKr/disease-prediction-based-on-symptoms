import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sqlite3
import os

# Import UI configuration
from ui_config import set_page_config_and_styles

# Set page config and apply styles
set_page_config_and_styles()

# --- Data Loading and Preprocessing (Cached) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("training.csv")
        tr = pd.read_csv("testing.csv")
    except FileNotFoundError:
        st.error("Error: 'training.csv' or 'testing.csv' not found. Please ensure they are in the same directory as app.py.")
        st.stop()

    # --- Column Name Cleaning for DataFrames ---
    # Strip whitespace, replace spaces with underscores, convert to lowercase, and fix double underscores
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower().str.replace('__', '_')
    tr.columns = tr.columns.str.strip().str.replace(' ', '_').str.lower().str.replace('__', '_')

    # List of symptoms - now in a consistent format (all lowercase, underscores)
    # This list is based on the cleaned column names observed from the CSVs.
    l1 = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
        'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
        'spotting_urination', # Standardized
        'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
        'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
        'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
        'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
        'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
        'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
        'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
        'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
        'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
        'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
        'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
        'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
        'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
        'foul_smell_of_urine',  # Standardized
        'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
        'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
        'abnormal_menstruation',
        'dischromic_patches',  # Standardized
        'watering_from_eyes', 'increased_appetite', 'polyuria',
        'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
        'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
        'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
        'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
        'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
        'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    ]
    l1 = [s.strip() for s in l1] 

    # Original disease list for display in the UI (no stripping/lower/underscores)
    original_disease_names = [
        'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
        'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
        'Gastroenteritis', 'Bronchial Asthma',
        'Hypertension', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)',
        'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
        'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis',
        'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heart attack',
        'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
        'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', # Original double space for display
        'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo'
    ]
    
    # Standardized disease list used for internal mapping (strip, replace spaces with underscores, lower)
    # This ensures consistency with cleaned 'prognosis' column values from CSVs.
    standardized_disease_names = [d.strip().replace('  ', ' ').replace(' ', '_').lower() for d in original_disease_names]

    # Clean 'prognosis' column values in DataFrames for consistency with mapping keys
    # Convert to string, strip whitespace, replace spaces with underscores, and convert to lowercase
    df['prognosis'] = df['prognosis'].astype(str).str.strip().str.replace(' ', '_').str.lower().str.replace('__', '_')
    tr['prognosis'] = tr['prognosis'].astype(str).str.strip().str.replace(' ', '_').str.lower().str.replace('__', '_')

    # Replace disease names with numerical labels using the standardized list
    disease_mapping = {d: i for i, d in enumerate(standardized_disease_names)}
    df.replace({'prognosis': disease_mapping}, inplace=True)
    tr.replace({'prognosis': disease_mapping}, inplace=True)
    
    # Check for any remaining non-numeric values in 'prognosis' after replacement
    non_numeric_df = df[pd.to_numeric(df['prognosis'], errors='coerce').isna()]
    non_numeric_tr = tr[pd.to_numeric(tr['prognosis'], errors='coerce').isna()]

    if not non_numeric_df.empty or not non_numeric_tr.empty:
        st.error("Error: Some disease names were not mapped to numbers. This indicates a mismatch between the disease list and CSV data.")
        st.write("Unmapped unique values in training data:")
        st.write(non_numeric_df['prognosis'].unique())
        st.write("Unmapped unique values in testing data:")
        st.write(non_numeric_tr['prognosis'].unique())
        st.stop() # Stop Streamlit execution if unmapped values are found

    # Use the cleaned column names for X and X_test
    X = df[l1]
    y = df["prognosis"].astype(np.int64).values
    X_test = tr[l1]
    y_test = tr["prognosis"].astype(np.int64).values

    # Return the original disease names for display purposes in the UI
    return l1, original_disease_names, X, y, X_test, y_test

l1, disease, X, y, X_test, y_test = load_data()

# --- Model Training (Cached) ---
@st.cache_resource
def train_decision_tree(X_data, y_data):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_data, np.ravel(y_data))
    return clf

@st.cache_resource
def train_random_forest(X_data, y_data):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_data, np.ravel(y_data))
    return clf

@st.cache_resource
def train_knn(X_data, y_data):
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_data, np.ravel(y_data))
    return knn

@st.cache_resource
def train_naive_bayes(X_data, y_data):
    gnb = GaussianNB()
    gnb.fit(X_data, np.ravel(y_data))
    return gnb

# Train all models
clf3 = train_decision_tree(X, y)
clf4 = train_random_forest(X, y)
knn = train_knn(X, y)
gnb = train_naive_bayes(X, y)

# --- Database Functions ---
def create_database_table(db_name, table_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(f"CREATE TABLE IF NOT EXISTS {table_name}(Name TEXT, Symtom1 TEXT, Symtom2 TEXT, Symtom3 TEXT, Symtom4 TEXT, Symtom5 TEXT, Disease TEXT)")
    conn.commit()
    conn.close()

def insert_prediction(db_name, table_name, name, s1, s2, s3, s4, s5, disease_pred):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(f"INSERT INTO {table_name}(Name, Symtom1, Symtom2, Symtom3, Symtom4, Symtom5, Disease) VALUES(?,?,?,?,?,?,?)",
              (name, s1, s2, s3, s4, s5, disease_pred))
    conn.commit()
    conn.close()

# Create tables if they don't exist
create_database_table('database.db', 'DecisionTree')
create_database_table('database.db', 'RandomForest')
create_database_table('database.db', 'NaiveBayes')
create_database_table('database1.db', 'KNearestNeighbour') # Separate DB for KNN as in notebook


# --- Plotting Functions ---
def scatterplt(disea_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    if disea_name and disea_name != "Not Found":
        # Find the original dataframe before replacement
        df_original = pd.read_csv("training.csv")
        # Apply the same cleaning to df_original columns for consistency
        df_original.columns = df_original.columns.str.strip().str.replace(' ', '_').str.lower().str.replace('__', '_')
        # And clean prognosis values for df_original
        df_original['prognosis'] = df_original['prognosis'].astype(str).str.strip().str.replace(' ', '_').str.lower().str.replace('__', '_')
        DF = df_original.set_index('prognosis')
        
        try:
            # Important: The disease name from 'disea_name' comes from the UI's display list ('disease' global variable)
            # which is the original_disease_names. For DF.loc, we need the standardized name.
            standardized_disea_name_for_loc = disea_name.strip().replace('  ', ' ').replace(' ', '_').lower()
            x_series = (DF.loc[standardized_disea_name_for_loc]).sum()
            x_series.drop(x_series[x_series==0].index, inplace=True)
            y_symptoms = x_series.keys()
            x_values = x_series.values
            ax.scatter(y_symptoms, x_values)
            ax.set_title(f"Symptoms for {disea_name}")
            ax.set_ylabel("Count")
            ax.set_xticks(range(len(y_symptoms)))
            ax.set_xticklabels(y_symptoms, rotation=90)
            st.pyplot(fig)
        except KeyError:
            st.info(f"No specific symptom data found for {disea_name} in the training dataset for plotting.")
    else:
        st.info("Cannot plot for 'Not Found' or empty disease prediction.")

def scatterinp(sym_list, y_values):
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_symptoms = [s for s in sym_list if s != "Select Here"]
    valid_y_values = [v for s, v in zip(sym_list, y_values) if s != "Select Here"]
    
    if valid_symptoms:
        ax.scatter(valid_symptoms, valid_y_values)
        ax.set_title("Input Symptoms")
        ax.set_ylabel("Presence (1 = Yes)")
        ax.set_xticks(range(len(valid_symptoms)))
        ax.set_xticklabels(valid_symptoms, rotation=90)
        st.pyplot(fig)
    else:
        st.info("No symptoms selected for input plot.")


# --- Prediction Functions ---
def make_prediction(model, model_name, patient_name, s1, s2, s3, s4, s5):
    if not patient_name:
        return "Please fill in patient name.", "warning"
    if s1 == "Select Here" or s2 == "Select Here":
        return "Please fill in at least the first two symptoms.", "warning"

    input_symptoms = [s1, s2, s3, s4, s5]
    l2 = [0] * len(l1)
    for i, symptom in enumerate(l1):
        if symptom in input_symptoms:
            l2[i] = 1

    try:
        predict = model.predict([l2])
        predicted_label = predict[0]

        # Use the original_disease_names (global 'disease' list) for display purposes
        predicted_disease_display = "Not Found"
        if 0 <= predicted_label < len(disease):
            predicted_disease_display = disease[predicted_label]
            
        # Insert into database using the display name (original format)
        db_name = 'database.db'
        if model_name == "KNearestNeighbour":
            db_name = 'database1.db'
        
        insert_prediction(db_name, model_name, patient_name, s1, s2, s3, s4, s5, predicted_disease_display)
        
        return predicted_disease_display, "success"

    except Exception as e:
        return f"Prediction error: {e}", "error"

def display_model_metrics(model, model_name):
    st.subheader(f"{model_name} Model Metrics")
    y_pred = model.predict(X_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Accuracy:")
        st.success(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
    with col2:
        st.write("Confusion Matrix:")
        st.code(confusion_matrix(y_test, y_pred))
    
    # The target_names in classification_report should match the labels the model was trained on
    # which are the standardized_disease_names.
    with st.expander(f"View {model_name} Classification Report"):
        st.code(classification_report(y_test, y_pred, target_names=[d.strip().replace('  ', ' ').replace(' ', '_').lower() for d in disease], zero_division=0))


# --- Streamlit UI Main Flow ---
st.markdown("<p class='main-header'>Disease Predictor using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>GitHub: JhaGauravKr</p>", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("Patient Information & Symptoms")
    patient_name = st.text_input("Name of the Patient *")

    st.write("Select Symptoms (at least first two are mandatory):")
    symptom1 = st.selectbox("Symptom 1 *", ["Select Here"] + sorted(l1), key="symptom1")
    symptom2 = st.selectbox("Symptom 2 *", ["Select Here"] + sorted(l1), key="symptom2")
    symptom3 = st.selectbox("Symptom 3", ["Select Here"] + sorted(l1), key="symptom3")
    symptom4 = st.selectbox("Symptom 4", ["Select Here"] + sorted(l1), key="symptom4")
    symptom5 = st.selectbox("Symptom 5", ["Select Here"] + sorted(l1), key="symptom5")

# --- Actions Section (Moved to Main Content) ---
st.markdown("---") # Separator
st.subheader("Actions")
col_actions = st.columns(3) # Use columns for buttons
with col_actions[0]:
    predict_button_all = st.button("Predict All Diseases")
with col_actions[1]:
    reset_button = st.button("Reset Inputs")
with col_actions[2]:
    exit_button = st.button("Exit System")
st.markdown("---") # Separator

# Main content area - Prediction Results
st.header("Prediction Results")

# Initialize session state for predictions if not already present
if 'pred_dt' not in st.session_state:
    st.session_state.pred_dt = None
if 'pred_rf' not in st.session_state:
    st.session_state.pred_rf = None
if 'pred_nb' not in st.session_state:
    st.session_state.pred_nb = None
if 'pred_knn' not in st.session_state:
    st.session_state.pred_knn = None
if 'final_outcome' not in st.session_state:
    st.session_state.final_outcome = None

if reset_button:
    st.session_state.pred_dt = None
    st.session_state.pred_rf = None
    st.session_state.pred_nb = None
    st.session_state.pred_knn = None
    st.session_state.final_outcome = None
    st.rerun() # Rerun to clear inputs in sidebar

if exit_button:
    st.stop() # Stops the app execution

if predict_button_all:
    # Clear previous predictions to avoid stale data if input changes without reset
    st.session_state.pred_dt = None
    st.session_state.pred_rf = None
    st.session_state.pred_nb = None
    st.session_state.pred_knn = None
    st.session_state.final_outcome = None

    # Run predictions with spinner
    with st.spinner("Predicting diseases..."):
        st.session_state.pred_dt, dt_status = make_prediction(clf3, "DecisionTree", patient_name, symptom1, symptom2, symptom3, symptom4, symptom5)
        st.session_state.pred_rf, rf_status = make_prediction(clf4, "RandomForest", patient_name, symptom1, symptom2, symptom3, symptom4, symptom5)
        st.session_state.pred_nb, nb_status = make_prediction(gnb, "NaiveBayes", patient_name, symptom1, symptom2, symptom3, symptom4, symptom5)
        st.session_state.pred_knn, knn_status = make_prediction(knn, "KNearestNeighbour", patient_name, symptom1, symptom2, symptom3, symptom4, symptom5)

    # Only proceed with final outcome if initial mandatory checks passed for at least one model
    if dt_status == "success" or rf_status == "success" or nb_status == "success" or knn_status == "success":
        # Determine final outcome based on majority vote
        outcomes = [st.session_state.pred_dt, st.session_state.pred_rf, st.session_state.pred_nb, st.session_state.pred_knn]
        outcome_count = {}
        for outcome in outcomes:
            # Exclude specific non-prediction strings for voting
            if outcome and outcome not in ["Not Found", "Prediction error", "Please fill in patient name.", "Please fill in at least the first two symptoms."]: 
                outcome_count[outcome] = outcome_count.get(outcome, 0) + 1

        majority_outcome = None
        for outcome, count in outcome_count.items():
            if count >= 2: # At least two models agree
                majority_outcome = outcome
                break
        
        if majority_outcome:
            st.session_state.final_outcome = majority_outcome
        else:
            # Check if any valid predictions were made but no majority
            if any(o and o not in ["Not Found", "Prediction error", "Please fill in patient name.", "Please fill in at least the first two symptoms."] for o in outcomes):
                st.session_state.final_outcome = "No clear majority. Please provide more symptoms if possible."
            else:
                 st.session_state.final_outcome = "Refill the symptoms or check for errors."
    else:
        # If any validation failed, set final outcome based on the first error encountered from make_prediction
        if dt_status == "warning": # Assuming warning means validation failed at the make_prediction start
            st.session_state.final_outcome = st.session_state.pred_dt


# Displaying results in a structured way
if st.session_state.pred_dt is not None: # Check if predictions have been attempted
    st.subheader("Individual Model Predictions")
    col_dt, col_rf, col_nb, col_knn = st.columns(4)

    # Use markdown with custom CSS classes for colored prediction boxes
    with col_dt:
        st.markdown(f"<div class='prediction-label bg-dt'><b>Decision Tree:</b> {st.session_state.pred_dt}</div>", unsafe_allow_html=True)
    with col_rf:
        st.markdown(f"<div class='prediction-label bg-rf'><b>Random Forest:</b> {st.session_state.pred_rf}</div>", unsafe_allow_html=True)
    with col_nb:
        st.markdown(f"<div class='prediction-label bg-nb'><b>Naive Bayes:</b> {st.session_state.pred_nb}</div>", unsafe_allow_html=True)
    with col_knn:
        st.markdown(f"<div class='prediction-label bg-knn'><b>KNN:</b> {st.session_state.pred_knn}</div>", unsafe_allow_html=True)

    if st.session_state.final_outcome:
        st.subheader("Final Outcome")
        # Determine the appropriate CSS class for the final outcome box based on its content
        final_outcome_css_class = "bg-final"
        if st.session_state.final_outcome in ["No clear majority. Please provide more symptoms if possible.", "Refill the symptoms or check for errors.", "Please fill in patient name.", "Please fill in at least the first two symptoms.", "Not Found"]:
            final_outcome_css_class = "bg-final-warning"
        elif st.session_state.final_outcome.startswith("Prediction error"):
            final_outcome_css_class = "bg-final-warning" # Or a dedicated error color if preferred

        st.markdown(f"<div class='prediction-label {final_outcome_css_class}'><b>Final Prediction:</b> {st.session_state.final_outcome}</div>", unsafe_allow_html=True)

        st.subheader("Symptom Plots")
        st.write("Below are scatter plots related to your input and the predicted disease (if applicable).")
        
        # Prepare data for input scatter plot
        selected_symptoms_for_plot = [s for s in [symptom1, symptom2, symptom3, symptom4, symptom5] if s != "Select Here"]
        y_input_plot_values = [1] * len(selected_symptoms_for_plot)
        scatterinp(selected_symptoms_for_plot, y_input_plot_values)

        # Plot for the final predicted disease
        # Only plot if a valid, positive prediction was made
        if st.session_state.final_outcome and st.session_state.final_outcome not in ["Not Found", "No clear majority. Please provide more symptoms if possible.", "Refill the symptoms or check for errors.", "Please fill in patient name.", "Please fill in at least the first two symptoms."]:
            scatterplt(st.session_state.final_outcome)
        else:
            st.info("No specific disease predicted with sufficient confidence for a detailed symptom plot.")

# Optional: Display model accuracies (can be expanded in an expander)
st.markdown("---")
st.header("Model Performance (on Test Data)")
with st.expander("Show Detailed Model Performance"):
    display_model_metrics(clf3, "DecisionTree")
    display_model_metrics(clf4, "RandomForest")
    display_model_metrics(gnb, "NaiveBayes")
    display_model_metrics(knn, "KNearestNeighbour")
