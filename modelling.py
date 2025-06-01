import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def modeling(X_train_path, X_test_path, y_train_path, y_test_path):
    # Load dataset yang sudah diprocessing
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze() 
    y_test = pd.read_csv(y_test_path).squeeze()
    
    print(f"Jumlah data X_train: {X_train.shape[0]}")
    print(f"Jumlah data X_test: {X_test.shape[0]}")
    print(f"Jumlah data y_train: {len(y_train)}")
    print(f"Jumlah data y_test: {len(y_test)}")
        
    # Cek Jumlah Kelas Unik Output (Status Gizi)
    num_classes = y_train.nunique()
    print(f"Jumlah kelas unik yang terdeteksi: {num_classes}")

    # Inisialisasi model XGBoost
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,    
        n_estimators=100,          
        learning_rate=0.1,        
        max_depth=5,              
        random_state=42,          
        n_jobs=-1,                  
        eval_metric='mlogloss'  
    )

    # Training Model
    print("\nMelatih model XGBoost...")
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    print("\n--- Evaluasi Model pada Test Set ---")
    print("Akurasi:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    # Path file hasil preprocessing dan split
    X_train_path = "data_balita_preprocessing/X_train.csv"
    X_test_path = "data_balita_preprocessing/X_test.csv"
    y_train_path = "data_balita_preprocessing/y_train.csv"
    y_test_path = "data_balita_preprocessing/y_test.csv"

    # Inisialisasi MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Stunting_Detection_XGBoost_Model")

    # Mengaktifkan autologging
    mlflow.xgboost.autolog()

    with mlflow.start_run(run_name="Modelling_XGBoost"):
        trained_model = modeling(X_train_path, X_test_path, y_train_path, y_test_path)