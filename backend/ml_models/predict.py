import pickle
import pandas as pd
import os

# Get the directory where this file is located
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model():
    """
    Load the trained model and label encoders
    """
    model_path = os.path.join(MODEL_DIR, 'salary_model.pkl')
    encoders_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)
    
    return model, label_encoders

def predict_salary(title, location, company):
    """
    Predict salary for a job based on title, location, and company
    
    Args:
        title (str): Job title (e.g., "Data Scientist")
        location (str): Location (e.g., "San Francisco")
        company (str): Company name (e.g., "Google")
    
    Returns:
        dict: Predicted salary and confidence
    """
    try:
        # Load model
        model, label_encoders = load_model()
        
        # Encode the input features (convert text to numbers)
        title_encoded = label_encoders['title'].transform([str(title)])[0]
        location_encoded = label_encoders['location'].transform([str(location)])[0]
        company_encoded = label_encoders['company'].transform([str(company)])[0]
        
        # Create input for prediction
        input_data = pd.DataFrame({
            'title_encoded': [title_encoded],
            'location_encoded': [location_encoded],
            'company_encoded': [company_encoded]
        })
        
        # Make prediction
        predicted_salary = model.predict(input_data)[0]
        
        return {
            "predicted_salary": round(predicted_salary, 2),
            "title": title,
            "location": location,
            "company": company
        }
    
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}"
        }

def get_model_stats():
    """
    Get statistics about the trained model
    """
    try:
        model, label_encoders = load_model()
        
        return {
            "model_type": "Random Forest Regressor",
            "features": ["title", "location", "company"],
            "num_trees": model.n_estimators,
            "trained_on_jobs": len(label_encoders['title'].classes_),
            "unique_titles": len(label_encoders['title'].classes_),
            "unique_locations": len(label_encoders['location'].classes_),
            "unique_companies": len(label_encoders['company'].classes_)
        }
    except Exception as e:
        return {
            "error": f"Could not load model stats: {str(e)}"
        }