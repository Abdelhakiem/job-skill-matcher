import os
import pickle
import json
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Base path of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PKL_PATH = os.path.join(BASE_DIR, 'rf_model.pkl')
DATA_PKL_PATH = os.path.join(BASE_DIR, 'rf_data.pkl')
SKILLS_CLUSTERS_PATH = os.path.join(BASE_DIR, '../data/processed/skills_clusters.json')


def initialize_model():
    global classifier, features_names, targets_names, skills_clusters_df
    with open(MODEL_PKL_PATH, "rb") as f:
        model = pickle.load(f)
    classifier = model['model_object']
    
    # Load metadata
    with open(DATA_PKL_PATH, "rb") as f:
        data = pickle.load(f)
    features_names = pd.Series(data['features_names'])
    targets_names = pd.Series(data['target_names'])

    # Load skill clusters
    with open(SKILLS_CLUSTERS_PATH, 'r') as f:
        skills_clusters = json.load(f)
    skills_clusters = pd.Series(skills_clusters)
    skills_clusters_df = pd.DataFrame(
        [(cluster, skill) for cluster, skills in skills_clusters.items() for skill in skills],
        columns=['cluster_id', 'skill']
    )


def get_skills_clusters(sample_skills, skills_clusters_df, features_names):
    """Convert list of skills to feature vector using skill clusters."""
    valid_skills = skills_clusters_df[skills_clusters_df['skill'].isin(sample_skills)]
    clusters_freq = valid_skills['cluster_id'].value_counts().reindex(features_names, fill_value=0)
    return clusters_freq.tolist()


def predict_job_probabilities(skills: list[str]) -> dict:
    """Takes a list of skills and returns sorted job probabilities."""
    if classifier is None:
        raise RuntimeError("Model is not initialized. Call `initialize_model()` first.")
    
    if not all(skill in skills_clusters_df['skill'].values for skill in skills):
        return {"error": "Invalid skills provided. Check spelling or data coverage."}
    
    cluster_vector = get_skills_clusters(skills, skills_clusters_df, features_names)
    input_df = pd.DataFrame([cluster_vector], columns=features_names)

    proba = classifier.predict_proba(input_df)
    positive_probs = [prob[0][1] for prob in proba]

    prediction_series = pd.Series(positive_probs, index=targets_names).sort_values(ascending=False)
    return prediction_series.to_dict()


    