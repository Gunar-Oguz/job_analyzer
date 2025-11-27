import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.path.append('..')
from database import get_jobs_from_db

def train_job_classifier():
    """
    Train a model to classify jobs into categories
    """
    print("Step 1: Fetching jobs from database...")
    jobs = get_jobs_from_db(limit=500)
    
    if not jobs:
        print("Error: No jobs found!")
        return
    
    print(f"Found {len(jobs)} jobs")
    
    # Convert to DataFrame
    df = pd.DataFrame(jobs)
    
    # Step 2: Create labels from job titles
    print("\nStep 2: Creating job categories...")
    
    def categorize_job(title):
        title_lower = title.lower()
        if 'machine learning' in title_lower or 'ml engineer' in title_lower:
            return 'ML Engineer'
        elif 'data scientist' in title_lower:
            return 'Data Scientist'
        elif 'data analyst' in title_lower or 'business analyst' in title_lower:
            return 'Data Analyst'
        elif 'data engineer' in title_lower:
            return 'Data Engineer'
        else:
            return 'Other'
    
    df['category'] = df['title'].apply(categorize_job)
    
    # Remove 'Other' category for cleaner training
    df = df[df['category'] != 'Other']
    
    print(f"Jobs with categories: {len(df)}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    
    if len(df) < 20:
        print("Error: Need more categorized jobs!")
        return
    
    # Step 3: Prepare text features
    print("\nStep 3: Converting text to numbers (TF-IDF)...")
    
    # Combine title and description for better features
    df['text'] = df['title'] + ' ' + df['description'].fillna('')
    
    # TF-IDF converts text to numbers
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['category']
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Step 4: Split data
    print("\nStep 4: Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} jobs")
    print(f"Test set: {X_test.shape[0]} jobs")
    
    # Step 5: Train classifier
    print("\nStep 5: Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Step 6: Evaluate
    print("\nStep 6: Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Step 7: Save model
    print("\nStep 7: Saving model...")
    with open('job_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("\n✓ Model saved as job_classifier.pkl")
    print("✓ Vectorizer saved as tfidf_vectorizer.pkl")
    
    return model, vectorizer

if __name__ == "__main__":
    train_job_classifier()