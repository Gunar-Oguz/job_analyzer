import pickle
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_classifier():
    """Load the trained classifier and vectorizer"""
    model_path = os.path.join(MODEL_DIR, 'job_classifier.pkl')
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def predict_job_category(title, description=""):
    """
    Predict job category from title and description
    
    Input: "Senior ML Engineer", "Build neural networks..."
    Output: "ML Engineer"
    """
    try:
        model, vectorizer = load_classifier()
        
        # Combine title + description (same as training)
        text = title + ' ' + description
        
        # Convert text to numbers
        X = vectorizer.transform([text])
        
        # Predict
        category = model.predict(X)[0]
        
        # Get probability scores
        probabilities = model.predict_proba(X)[0]
        categories = model.classes_
        
        # Create confidence dict
        confidence = {cat: round(prob * 100, 1) 
                      for cat, prob in zip(categories, probabilities)}
        
        return {
            "predicted_category": category,
            "confidence": confidence,
            "title": title
        }
    
    except Exception as e:
        return {"error": str(e)}