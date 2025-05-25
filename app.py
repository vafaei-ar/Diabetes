import gradio as gr
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier # Assuming RF, add others if needed

# Define categorical mappings
SEX_MAP = {'Female': 0, 'Male': 1}
ETHNICITY_MAP = {'Not Hispanic or Latino': 0, 'Hispanic or Latino': 1}
MARITAL_STATUS_MAP = {'Single': 0, 'Married': 1}
RACE_MAP = {
    'White': 0,
    'Asian': 1,
    'American Indian or Alaska Native': 2,
    'Native Hawaiian or Other Pacific Islander': 3,
    'Black or African American': 4,
    'Other Race': 5,
    'Unknown': 6
}

# Invert maps for display in dropdowns if necessary (or use keys directly)
RACE_CHOICES = list(RACE_MAP.keys())
SEX_CHOICES = list(SEX_MAP.keys())
ETHNICITY_CHOICES = list(ETHNICITY_MAP.keys())
MARITAL_STATUS_CHOICES = list(MARITAL_STATUS_MAP.keys())

MODEL_DIR = "models"

def get_available_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR) # Create models directory if it doesn't exist
        return ["No models found. Please add .joblib models to the 'models' directory."]
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
    if not models:
        return ["No models found. Please add .joblib models to the 'models' directory."]
    return models

# Define all features in the order your model expects them
# IMPORTANT: This order must match the training data
EXPECTED_COLUMNS = [
    'sex', 'race', 'ethnicity', 'marital_status', 'Prior_Mean_Glu',
    'PT_ELX_GRP_1', 'PT_ELX_GRP_2', 'PT_ELX_GRP_3', 'PT_ELX_GRP_4',
    'PT_ELX_GRP_5', 'PT_ELX_GRP_6', 'PT_ELX_GRP_7', 'PT_ELX_GRP_8',
    'PT_ELX_GRP_9', 'PT_ELX_GRP_10', 'PT_ELX_GRP_13', 'PT_ELX_GRP_14',
    'PT_ELX_GRP_15', 'PT_ELX_GRP_16', 'PT_ELX_GRP_17', 'PT_ELX_GRP_18',
    'PT_ELX_GRP_19', 'PT_ELX_GRP_20', 'PT_ELX_GRP_21', 'PT_ELX_GRP_22',
    'PT_ELX_GRP_23', 'PT_ELX_GRP_24', 'PT_ELX_GRP_25', 'PT_ELX_GRP_26',
    'PT_ELX_GRP_27', 'PT_ELX_GRP_28', 'PT_ELX_GRP_29', 'PT_ELX_GRP_30',
    'PT_ELX_GRP_31', 'MOF', 'SDOH', 'Gallstone', 'acei_drug', 'statin_drug',
    'diuretic_drug', 'antiplatelet_drug', 'anticoagulant_drug',
    'nsaid_drug', 'ppi_drug', 'beta_blokers_drug', 'vasodilators_drug',
    'caaa_drug', 'ccb_drug', 'paaab_drug', 'age', 'BMI', 'Body_weight',
    'SBP', 'DBP', 'Mean_AST', 'Mean_ALT', 'Mean_TBIL', 'Mean_ALP',
    'Mean_Hgb', 'Mean_HCT', 'Mean_Cr', 'Mean_PLT', 'Mean_WBC', 'Mean_BUN',
    'Mean_AGAP', 'Mean_Protein', 'Smoking', 'eGFR', 'ED_visits', 'LOS',
    'Prediabetes', 'Alcohol_use', 'Famly_hist_diabetes', 'NAFLD',
    'Hist_Gesta_diabetes', 'Pregnancy', 'numof_med_visits',
    'History_AP_necrosis', 'Necrosectomy', 'Steroids_drugs',
    'oral_contraceptive', 'cholelithiasis', 'acute_cholecystitis',
    'hypertriglyceridemia'
]

def predict_diabetes(model_name, sex, race, ethnicity, marital_status, Prior_Mean_Glu,
       PT_ELX_GRP_1, PT_ELX_GRP_2, PT_ELX_GRP_3, PT_ELX_GRP_4,
       PT_ELX_GRP_5, PT_ELX_GRP_6, PT_ELX_GRP_7, PT_ELX_GRP_8,
       PT_ELX_GRP_9, PT_ELX_GRP_10, PT_ELX_GRP_13, PT_ELX_GRP_14,
       PT_ELX_GRP_15, PT_ELX_GRP_16, PT_ELX_GRP_17, PT_ELX_GRP_18,
       PT_ELX_GRP_19, PT_ELX_GRP_20, PT_ELX_GRP_21, PT_ELX_GRP_22,
       PT_ELX_GRP_23, PT_ELX_GRP_24, PT_ELX_GRP_25, PT_ELX_GRP_26,
       PT_ELX_GRP_27, PT_ELX_GRP_28, PT_ELX_GRP_29, PT_ELX_GRP_30,
       PT_ELX_GRP_31, MOF, SDOH, Gallstone, acei_drug, statin_drug,
       diuretic_drug, antiplatelet_drug, anticoagulant_drug,
       nsaid_drug, ppi_drug, beta_blokers_drug, vasodilators_drug,
       caaa_drug, ccb_drug, paaab_drug, age, BMI, Body_weight,
       SBP, DBP, Mean_AST, Mean_ALT, Mean_TBIL, Mean_ALP,
       Mean_Hgb, Mean_HCT, Mean_Cr, Mean_PLT, Mean_WBC, Mean_BUN,
       Mean_AGAP, Mean_Protein, Smoking, eGFR, ED_visits, LOS,
       Prediabetes, Alcohol_use, Famly_hist_diabetes, NAFLD,
       Hist_Gesta_diabetes, Pregnancy, numof_med_visits,
       History_AP_necrosis, Necrosectomy, Steroids_drugs,
       oral_contraceptive, cholelithiasis, acute_cholecystitis,
       hypertriglyceridemia):

    if not model_name or "No models found" in model_name:
        return "Please select a valid model from the 'models/' directory."

    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return f"Model file {model_name} not found in {MODEL_DIR}."

    try:
        model = joblib.load(model_path)
    except Exception as e:
        return f"Error loading model: {e}"

    # Prepare data for prediction
    input_data = {
        'sex': SEX_MAP[sex],
        'race': RACE_MAP[race],
        'ethnicity': ETHNICITY_MAP[ethnicity],
        'marital_status': MARITAL_STATUS_MAP[marital_status],
        'Prior_Mean_Glu': float(Prior_Mean_Glu),
        'PT_ELX_GRP_1': float(PT_ELX_GRP_1), 'PT_ELX_GRP_2': float(PT_ELX_GRP_2), 'PT_ELX_GRP_3': float(PT_ELX_GRP_3),
        'PT_ELX_GRP_4': float(PT_ELX_GRP_4), 'PT_ELX_GRP_5': float(PT_ELX_GRP_5), 'PT_ELX_GRP_6': float(PT_ELX_GRP_6),
        'PT_ELX_GRP_7': float(PT_ELX_GRP_7), 'PT_ELX_GRP_8': float(PT_ELX_GRP_8), 'PT_ELX_GRP_9': float(PT_ELX_GRP_9),
        'PT_ELX_GRP_10': float(PT_ELX_GRP_10), 'PT_ELX_GRP_13': float(PT_ELX_GRP_13), 'PT_ELX_GRP_14': float(PT_ELX_GRP_14),
        'PT_ELX_GRP_15': float(PT_ELX_GRP_15), 'PT_ELX_GRP_16': float(PT_ELX_GRP_16), 'PT_ELX_GRP_17': float(PT_ELX_GRP_17),
        'PT_ELX_GRP_18': float(PT_ELX_GRP_18), 'PT_ELX_GRP_19': float(PT_ELX_GRP_19), 'PT_ELX_GRP_20': float(PT_ELX_GRP_20),
        'PT_ELX_GRP_21': float(PT_ELX_GRP_21), 'PT_ELX_GRP_22': float(PT_ELX_GRP_22), 'PT_ELX_GRP_23': float(PT_ELX_GRP_23),
        'PT_ELX_GRP_24': float(PT_ELX_GRP_24), 'PT_ELX_GRP_25': float(PT_ELX_GRP_25), 'PT_ELX_GRP_26': float(PT_ELX_GRP_26),
        'PT_ELX_GRP_27': float(PT_ELX_GRP_27), 'PT_ELX_GRP_28': float(PT_ELX_GRP_28), 'PT_ELX_GRP_29': float(PT_ELX_GRP_29),
        'PT_ELX_GRP_30': float(PT_ELX_GRP_30), 'PT_ELX_GRP_31': float(PT_ELX_GRP_31),
        'MOF': float(MOF), 'SDOH': float(SDOH), 'Gallstone': float(Gallstone),
        'acei_drug': float(acei_drug), 'statin_drug': float(statin_drug), 'diuretic_drug': float(diuretic_drug),
        'antiplatelet_drug': float(antiplatelet_drug), 'anticoagulant_drug': float(anticoagulant_drug),
        'nsaid_drug': float(nsaid_drug), 'ppi_drug': float(ppi_drug), 'beta_blokers_drug': float(beta_blokers_drug),
        'vasodilators_drug': float(vasodilators_drug), 'caaa_drug': float(caaa_drug), 'ccb_drug': float(ccb_drug),
        'paaab_drug': float(paaab_drug), 'age': float(age), 'BMI': float(BMI), 'Body_weight': float(Body_weight),
        'SBP': float(SBP), 'DBP': float(DBP), 'Mean_AST': float(Mean_AST), 'Mean_ALT': float(Mean_ALT),
        'Mean_TBIL': float(Mean_TBIL), 'Mean_ALP': float(Mean_ALP), 'Mean_Hgb': float(Mean_Hgb),
        'Mean_HCT': float(Mean_HCT), 'Mean_Cr': float(Mean_Cr), 'Mean_PLT': float(Mean_PLT),
        'Mean_WBC': float(Mean_WBC), 'Mean_BUN': float(Mean_BUN), 'Mean_AGAP': float(Mean_AGAP),
        'Mean_Protein': float(Mean_Protein), 'Smoking': float(Smoking), 'eGFR': float(eGFR),
        'ED_visits': float(ED_visits), 'LOS': float(LOS), 'Prediabetes': float(Prediabetes),
        'Alcohol_use': float(Alcohol_use), 'Famly_hist_diabetes': float(Famly_hist_diabetes),
        'NAFLD': float(NAFLD), 'Hist_Gesta_diabetes': float(Hist_Gesta_diabetes), 'Pregnancy': float(Pregnancy),
        'numof_med_visits': float(numof_med_visits), 'History_AP_necrosis': float(History_AP_necrosis),
        'Necrosectomy': float(Necrosectomy), 'Steroids_drugs': float(Steroids_drugs),
        'oral_contraceptive': float(oral_contraceptive), 'cholelithiasis': float(cholelithiasis),
        'acute_cholecystitis': float(acute_cholecystitis), 'hypertriglyceridemia': float(hypertriglyceridemia)
    }
    
    # Create DataFrame in the correct order
    try:
        df = pd.DataFrame([input_data], columns=EXPECTED_COLUMNS)
    except Exception as e:
        return f"Error creating DataFrame: {e}. Check EXPECTED_COLUMNS and input_data keys."

    # Make prediction
    try:
        prediction = model.predict(df)
        # You might need to access the first element if prediction is an array
        # e.g., result = prediction[0] 
        # Also, convert to a more human-readable output
        result = prediction[0] 
        if result == 1:
            return "Prediction: Positive for Diabetes"
        else:
            return "Prediction: Negative for Diabetes"
    except Exception as e:
        return f"Error during prediction: {e}"

# Define Gradio inputs
inputs = [
    gr.Dropdown(choices=get_available_models(), label="Select Model"),
    gr.Dropdown(choices=SEX_CHOICES, label="Sex"),
    gr.Dropdown(choices=RACE_CHOICES, label="Race"),
    gr.Dropdown(choices=ETHNICITY_CHOICES, label="Ethnicity"),
    gr.Dropdown(choices=MARITAL_STATUS_CHOICES, label="Marital Status"),
    gr.Number(label="Prior Mean Glu"),
    gr.Number(label="PT_ELX_GRP_1"), gr.Number(label="PT_ELX_GRP_2"), gr.Number(label="PT_ELX_GRP_3"),
    gr.Number(label="PT_ELX_GRP_4"), gr.Number(label="PT_ELX_GRP_5"), gr.Number(label="PT_ELX_GRP_6"),
    gr.Number(label="PT_ELX_GRP_7"), gr.Number(label="PT_ELX_GRP_8"), gr.Number(label="PT_ELX_GRP_9"),
    gr.Number(label="PT_ELX_GRP_10"), gr.Number(label="PT_ELX_GRP_13"), gr.Number(label="PT_ELX_GRP_14"),
    gr.Number(label="PT_ELX_GRP_15"), gr.Number(label="PT_ELX_GRP_16"), gr.Number(label="PT_ELX_GRP_17"),
    gr.Number(label="PT_ELX_GRP_18"), gr.Number(label="PT_ELX_GRP_19"), gr.Number(label="PT_ELX_GRP_20"),
    gr.Number(label="PT_ELX_GRP_21"), gr.Number(label="PT_ELX_GRP_22"), gr.Number(label="PT_ELX_GRP_23"),
    gr.Number(label="PT_ELX_GRP_24"), gr.Number(label="PT_ELX_GRP_25"), gr.Number(label="PT_ELX_GRP_26"),
    gr.Number(label="PT_ELX_GRP_27"), gr.Number(label="PT_ELX_GRP_28"), gr.Number(label="PT_ELX_GRP_29"),
    gr.Number(label="PT_ELX_GRP_30"), gr.Number(label="PT_ELX_GRP_31"),
    gr.Number(label="MOF"), gr.Number(label="SDOH"), gr.Number(label="Gallstone"),
    gr.Number(label="ACE Inhibitor Drug"), gr.Number(label="Statin Drug"), gr.Number(label="Diuretic Drug"),
    gr.Number(label="Antiplatelet Drug"), gr.Number(label="Anticoagulant Drug"),
    gr.Number(label="NSAID Drug"), gr.Number(label="PPI Drug"), gr.Number(label="Beta Blockers Drug"),
    gr.Number(label="Vasodilators Drug"), gr.Number(label="CAAA Drug"), gr.Number(label="CCB Drug"),
    gr.Number(label="PAAAB Drug"), gr.Number(label="Age"), gr.Number(label="BMI"),
    gr.Number(label="Body Weight (kg)"), gr.Number(label="SBP (Systolic Blood Pressure)"),
    gr.Number(label="DBP (Diastolic Blood Pressure)"), gr.Number(label="Mean AST"), gr.Number(label="Mean ALT"),
    gr.Number(label="Mean TBIL"), gr.Number(label="Mean ALP"), gr.Number(label="Mean Hgb"),
    gr.Number(label="Mean HCT"), gr.Number(label="Mean Cr"), gr.Number(label="Mean PLT"),
    gr.Number(label="Mean WBC"), gr.Number(label="Mean BUN"), gr.Number(label="Mean AGAP"),
    gr.Number(label="Mean Protein"), gr.Number(label="Smoking"), gr.Number(label="eGFR"),
    gr.Number(label="ED Visits"), gr.Number(label="LOS (Length of Stay)"), gr.Number(label="Prediabetes"),
    gr.Number(label="Alcohol Use"), gr.Number(label="Family History of Diabetes"),
    gr.Number(label="NAFLD"), gr.Number(label="History of Gestational Diabetes"),
    gr.Number(label="Pregnancy"), gr.Number(label="Number of Medical Visits"),
    gr.Number(label="History AP Necrosis"), gr.Number(label="Necrosectomy"),
    gr.Number(label="Steroids Drugs"), gr.Number(label="Oral Contraceptive"),
    gr.Number(label="Cholelithiasis"), gr.Number(label="Acute Cholecystitis"),
    gr.Number(label="Hypertriglyceridemia")
]

# Define output
output = gr.Textbox(label="Prediction Result")

# Create and launch the Gradio interface
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs=output,
    title="Diabetes Prediction",
    description="Enter patient data to predict diabetes. Ensure your models are in the 'models' directory.",
    allow_flagging='never'
)

if __name__ == "__main__":
    iface.launch()
