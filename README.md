# Diabetes
The Diabetes Project.

## Diabetes Prediction Gradio Application

This application allows you to predict the likelihood of diabetes based on various patient inputs using trained scikit-learn models.

### Prerequisites

- Python 3.7+
- Pip (Python package installer)

### Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add your trained models:**
    - Place your trained scikit-learn model files (which should be saved with the `.joblib` extension) into the `models/` directory.
    - For example, if you have a model named `my_model.joblib`, place it like so: `models/my_model.joblib`.
    - The application will automatically detect models in this directory. See `models/README.md` for more details.

### Running the Application

1.  **Ensure your virtual environment is activated and you are in the project's root directory.**
2.  **Run the Gradio app:**
    ```bash
    python app.py
    ```
3.  Open your web browser and navigate to the URL provided by Gradio (usually `http://127.0.0.1:7860` or `http://localhost:7860`).

### Input Features

The model requires the following input features:

*   **Categorical (select from dropdown):**
    *   `sex`
    *   `race`
    *   `ethnicity`
    *   `marital_status`
*   **Numerical (enter value):**
    *   `Prior_Mean_Glu`
    *   `PT_ELX_GRP_1` through `PT_ELX_GRP_31` (Elixhauser Comorbidity Index groups)
    *   `MOF` (Multiorgan Failure)
    *   `SDOH` (Social Determinants of Health)
    *   `Gallstone`
    *   `acei_drug`, `statin_drug`, ..., `paaab_drug` (various medication usages)
    *   `age`
    *   `BMI`
    *   `Body_weight`
    *   `SBP`, `DBP` (Systolic/Diastolic Blood Pressure)
    *   `Mean_AST`, `Mean_ALT`, ..., `Mean_Protein` (various lab values)
    *   `Smoking`
    *   `eGFR` (Estimated Glomerular Filtration Rate)
    *   `ED_visits`
    *   `LOS` (Length of Stay)
    *   `Prediabetes`
    *   `Alcohol_use`
    *   `Famly_hist_diabetes` (Family history of diabetes)
    *   `NAFLD` (Non-alcoholic Fatty Liver Disease)
    *   `Hist_Gesta_diabetes` (History of Gestational Diabetes)
    *   `Pregnancy`
    *   `numof_med_visits` (Number of medical visits)
    *   `History_AP_necrosis` (History of Acute Pancreatitis with necrosis)
    *   `Necrosectomy`
    *   `Steroids_drugs`
    *   `oral_contraceptive`
    *   `cholelithiasis`
    *   `acute_cholecystitis`
    *   `hypertriglyceridemia`

**Important:** The order and type of features are critical. The `app.py` script handles the necessary conversions for categorical features. Ensure numerical features are provided as appropriate numbers. The model(s) you place in the `models` directory must have been trained on features in the same order as defined in `EXPECTED_COLUMNS` within `app.py`.
