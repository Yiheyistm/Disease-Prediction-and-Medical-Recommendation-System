from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16) 

# Load datasets
sym_des = pd.read_csv("kaggle_dataset/symptoms_df.csv")
precautions = pd.read_csv("kaggle_dataset/precautions_df.csv")
workout = pd.read_csv("kaggle_dataset/workout_df.csv")
description = pd.read_csv("kaggle_dataset/description.csv")
medications = pd.read_csv('kaggle_dataset/medications.csv')
diets = pd.read_csv("kaggle_dataset/diets.csv")

train_data = pd.read_csv('kaggle_dataset/Training.csv')
test_data = pd.read_csv('kaggle_dataset/Testing.csv')

top_diseases = train_data['prognosis'].value_counts().head(20).index.tolist()
all_symptoms = [col.replace('_', ' ').lower() for col in train_data.columns if col != 'prognosis']

symptom_counts = train_data.drop('prognosis', axis=1).sum().sort_values(ascending=False)
top_symptoms = [symptom.replace('_', ' ').lower() for symptom in symptom_counts.head(20).index.tolist()]

symptoms_list = {symptom: idx for idx, symptom in enumerate(train_data.columns[:-1])}
diseases_list = {idx: disease for idx, disease in enumerate(train_data['prognosis'].unique())}
symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()}

# Model training
def train_models():
    X = train_data.drop('prognosis', axis=1)
    y = train_data['prognosis']
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(kernel='linear', probability=True),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models, model_metrics = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        trained_models[name] = model
        model_metrics[name] = {'accuracy': accuracy, 'report': report}
    
    return trained_models, model_metrics

model_path = 'model/RandomForest.pkl'
if os.path.exists(model_path):
    best_model = pickle.load(open(model_path, 'rb'))
    _, model_metrics = train_models()
else:
    trained_models, model_metrics = train_models()
    best_model = trained_models['RandomForest']
    os.makedirs('model', exist_ok=True)
    pickle.dump(best_model, open(model_path, 'wb'))

def information(predicted_dis):
    disease_description = description[description['Disease'] == predicted_dis]['Description']
    disease_description = " ".join([w for w in disease_description])
    
    disease_precautions = precautions[precautions['Disease'] == predicted_dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    disease_precautions = [col for col in disease_precautions.values]
    
    disease_medications = medications[medications['Disease'] == predicted_dis]['Medication']
    disease_medications = [med for med in disease_medications.values]
    
    disease_diet = diets[diets['Disease'] == predicted_dis]['Diet']
    disease_diet = [die for die in disease_diet.values]
    
    disease_workout = workout[workout['disease'] == predicted_dis]['workout']
    
    return disease_description, disease_precautions, disease_medications, disease_diet, disease_workout

def predicted_value(patient_symptoms):
    i_vector = np.zeros(len(symptoms_list_processed))
    for i in patient_symptoms:
        i_vector[symptoms_list_processed[i]] = 1
    return diseases_list[best_model.predict([i_vector])[0]]

def correct_spelling(symptom):
    closest_match, score = process.extractOne(symptom, symptoms_list_processed.keys())
    if score >= 80:
        return closest_match
    return None

def get_model_analysis_plot():
    plt.figure(figsize=(10, 6))
    accuracies = [metrics['accuracy'] for metrics in model_metrics.values()]
    plt.bar(model_metrics.keys(), accuracies)
    plt.title('Model Comparison - Accuracy Scores')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get symptoms from checkboxes
        checkbox_symptoms = request.form.getlist('symptoms')
        
        # Get symptoms from input field
        input_field = request.form.get('symptomsInput', '')
        typed_symptoms = [s.strip().lower() for s in input_field.split(',') if s.strip()]

        # Combine both sources
        raw_symptoms = list(set([s.lower() for s in checkbox_symptoms] + typed_symptoms))

        if not raw_symptoms:
            message = "Please select or enter at least one valid symptom."
            return render_template('index.html', message=message,
                                   all_symptoms=all_symptoms, 
                                   top_diseases=top_diseases,
                                   top_symptoms=top_symptoms)

        # Correct the spelling of each symptom
        corrected_symptoms = []
        for symptom in raw_symptoms:
            corrected_symptom = correct_spelling(symptom)
            if corrected_symptom:
                corrected_symptoms.append(corrected_symptom)
            else:
                message = f"Symptom '{symptom}' not found in the database."
                return render_template('index.html', message=message,
                                       all_symptoms=all_symptoms, 
                                       top_diseases=top_diseases,
                                       top_symptoms=top_symptoms)

        # Predict disease
        predicted_disease = predicted_value(corrected_symptoms)
        dis_des, precautions_list, medications_list, rec_diet_list, workout = information(predicted_disease)

        my_precautions = [i for i in precautions_list[0]]
        medications = ast.literal_eval(medications_list[0]) if medications_list else []
        rec_diet = ast.literal_eval(rec_diet_list[0]) if rec_diet_list else []

        # Store prediction result in session for /analysis
        session['prediction'] = {
            'predicted_disease': predicted_disease,
            'corrected_symptoms': corrected_symptoms,
            'top_predictions': None,  # optional if you're showing probabilities later
            'description': dis_des,
            'precautions': my_precautions,
            'medications': medications,
            'diet': rec_diet,
            'workout': list(workout)  # convert series to list if needed
        }

        return render_template('index.html',
                               symptoms=corrected_symptoms, 
                               predicted_disease=predicted_disease, 
                               dis_des=dis_des,
                               my_precautions=my_precautions, 
                               medications=medications, 
                               my_diet=rec_diet,
                               workout=workout,
                               all_symptoms=all_symptoms,
                               top_diseases=top_diseases,
                               top_symptoms=top_symptoms)

    return render_template('index.html',
                           all_symptoms=all_symptoms, 
                           top_diseases=top_diseases,
                           top_symptoms=top_symptoms)


@app.route('/analysis')
def analysis():
    # Get prediction data from session
    prediction_data = session.get('prediction', {})
    
    if not prediction_data:
        return redirect('/')  # Redirect to home if no prediction data in session

    # Get symptoms and disease from prediction data
    corrected = prediction_data.get('corrected_symptoms', [])
    predicted_disease = prediction_data.get('predicted_disease')

    # Create input vector
    i_vector = np.zeros(len(symptoms_list_processed))
    for symptom in corrected:
        i_vector[symptoms_list_processed[symptom]] = 1

    # Get prediction probabilities
    prediction_prob = best_model.predict_proba([i_vector])[0]

    # Show top 5 probable diseases
    top_predictions = sorted(list(enumerate(prediction_prob)), key=lambda x: x[1], reverse=True)[:5]
    top_predictions = [(diseases_list[idx], round(prob * 100, 2)) for idx, prob in top_predictions]

    # Optional: Feature importance
    if hasattr(best_model, "feature_importances_"):
        feat_imp = best_model.feature_importances_
        top_indices = np.argsort(feat_imp)[-10:][::-1]
        top_features = [(list(symptoms_list_processed.keys())[i], round(feat_imp[i], 4)) for i in top_indices]
    else:
        top_features = []

    return render_template('analysis.html',
                          corrected_symptoms=corrected,
                          predicted_disease=predicted_disease,
                          top_predictions=top_predictions,
                          triggered_symptoms=corrected,
                          top_features=top_features,
                          description=prediction_data.get('description'),
                          precautions=prediction_data.get('precautions'),
                          medications=prediction_data.get('medications'),
                          diet=prediction_data.get('diet'),
                          workout=prediction_data.get('workout'))


@app.route('/model_analysis')
def model_analysis():
    # Model accuracy plot
    accuracy_plot = get_model_analysis_plot()

    # Top diseases from training data
    top_disease_counts = train_data['prognosis'].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_disease_counts.values, y=top_disease_counts.index, palette='rocket')
    plt.title('Top 10 Diseases in Training Data')
    plt.xlabel('Number of Records')
    plt.tight_layout()
    disease_img = io.BytesIO()
    plt.savefig(disease_img, format='png')
    disease_img.seek(0)
    disease_plot = base64.b64encode(disease_img.getvalue()).decode()
    plt.close()

    # Top symptoms from training data
    symptom_counts = train_data.drop('prognosis', axis=1).sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=symptom_counts.values, y=[s.replace('_', ' ') for s in symptom_counts.index], palette='mako')
    plt.title('Top 10 Reported Symptoms')
    plt.xlabel('Frequency')
    plt.tight_layout()
    symptom_img = io.BytesIO()
    plt.savefig(symptom_img, format='png')
    symptom_img.seek(0)
    symptom_plot = base64.b64encode(symptom_img.getvalue()).decode()
    plt.close()

    return render_template('view_model_analysis.html', 
                           accuracy_plot=accuracy_plot, 
                           disease_plot=disease_plot,
                           symptom_plot=symptom_plot,
                           model_metrics=model_metrics)

@app.route('/')
def index():
    return render_template('index.html', all_symptoms=all_symptoms, top_diseases=top_diseases, top_symptoms=top_symptoms)

if __name__ == '__main__':
    app.run(debug=True)
