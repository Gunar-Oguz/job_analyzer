import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('..')
from database import get_jobs_from_db

def train_salary_model():
    """
    Train a Random Forest model to predict salaries
    """
    print("Step 1: Fetching jobs from database...")
    jobs = get_jobs_from_db(limit=500)
    
    if not jobs:
        print("Error: No jobs found in database!")
        return
    
    print(f"Found {len(jobs)} jobs")
    
    # Convert to DataFrame
    df = pd.DataFrame(jobs)
    
    # Step 2: Filter jobs with salary data
    print("\nStep 2: Filtering jobs with salary data...")
    df = df[df['salary_min'].notna() & (df['salary_min'] > 0)]
    print(f"Jobs with salary data: {len(df)}")
    
    if len(df) < 50:
        print("Error: Need at least 50 jobs with salary data!")
        return
    
    # Step 3: Create target variable (what we want to predict)
    print("\nStep 3: Creating target variable (average salary)...")
    df['salary_avg'] = (df['salary_min'] + df['salary_max']) / 2
    
    # Step 4: Select features (inputs to predict from)
    print("\nStep 4: Preparing features...")
    features = ['title', 'location', 'company']
    
    # Remove rows with missing features
    df_clean = df[features + ['salary_avg']].dropna()
    print(f"Clean data: {len(df_clean)} jobs")
    
    # Step 5: Encode text to numbers (ML needs numbers!)
    print("\nStep 5: Converting text to numbers...")
    label_encoders = {}
    
    for feature in features:
        le = LabelEncoder()
        df_clean[feature + '_encoded'] = le.fit_transform(df_clean[feature].astype(str))
        label_encoders[feature] = le
    
    # Step 6: Prepare training data
    print("\nStep 6: Splitting data into training and test sets...")
    X = df_clean[['title_encoded', 'location_encoded', 'company_encoded']]
    y = df_clean['salary_avg']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} jobs")
    print(f"Test set: {len(X_test)} jobs")
    
    # Step 7: Train the model
    print("\nStep 7: Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Step 8: Evaluate accuracy
    print("\nStep 8: Testing model accuracy...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.2%}")
    print(f"Test accuracy: {test_score:.2%}")
    
    # Step 9: Save the model
    print("\nStep 9: Saving model...")
    with open('salary_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    print("\n✓ Model saved successfully!")
    print(f"✓ Model can predict salaries with {test_score:.1%} accuracy")
    
    return model, label_encoders

if __name__ == "__main__":
    train_salary_model()
    