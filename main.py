from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, Field, field_validator
import joblib
import os
import uvicorn
import io
import requests
import zipfile
from sklearn.tree import DecisionTreeClassifier
from google.cloud import storage


MODEL_URL = "https://storage.googleapis.com/fast_api_crop_production/best_model.zip"
TARGET_ENCODINGS_URL = "https://storage.googleapis.com/fast_api_crop_production/target_encodings.pkl"
LOW_IMPORTANCE_FEATURES_URL = "https://storage.googleapis.com/fast_api_crop_production/low_importance_features.pkl"


global_model = None
global_target_encodings = None
global_low_importance_features = None

MODEL_FILENAME = "best_model.pkl"
MODEL_ZIP = "best_model.zip"

app = FastAPI(
    title="Crop Production Prediction API",
    description="API for predicting crop production using a Random Forest model",
    version="1.0.0")




import json
import os
from google.cloud import storage

def download_model_from_gcs():
    """Download the model from Google Cloud Storage (GCS) if it does not exist locally."""
    try:
        model_zip = os.path.join(os.getcwd(), "best_model.zip")

        if os.path.exists(model_zip):
            print(f"File {model_zip} already exists. Skipping download.")
            return True

        # Load credentials from Render environment variable
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not credentials_json:
            raise ValueError("GCS credentials not found. Ensure GOOGLE_APPLICATION_CREDENTIALS_JSON is set.")

        credentials_dict = json.loads(credentials_json)

        # Save credentials to a temporary file
        temp_credentials_path = "/tmp/gcs_credentials.json"
        with open(temp_credentials_path, "w") as temp_file:
            json.dump(credentials_dict, temp_file)

        # Set environment variable for Google Cloud authentication
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials_path

        # Define your GCS bucket and model path
        BUCKET_NAME = "fast_api_crop_production"  # Replace with your actual bucket name
        SOURCE_BLOB_NAME = "best_model.zip"  # Replace with the path in GCS

        # Initialize the Google Cloud Storage client
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(SOURCE_BLOB_NAME)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_zip), exist_ok=True)

        # Download the model from GCS
        blob.download_to_filename(model_zip)

        print(f"Successfully downloaded {model_zip} from GCS.")
        return True

    except Exception as e:
        print(f"Error downloading model from GCS: {e}")
        return False

    


def extract_and_load_model():
    """
    Extracts bestmodel.pkl from the ZIP file (if not already extracted) and loads it into memory.
    """
    global global_model


     # If the model file already exists, load it without extracting again
    if os.path.exists(MODEL_FILENAME):
        try:
            # Load the model using joblib (since the model might have been saved using joblib)
            global_model = joblib.load(MODEL_FILENAME)
            print("Model loaded successfully from the local file.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # If the model file does not exist, check for the ZIP file
    if os.path.exists(MODEL_ZIP):
        try:
            with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
                zip_ref.extractall(".")
            
            # Ensure extraction was successful before loading
            if os.path.exists(MODEL_FILENAME):  
                global_model = joblib.load(MODEL_FILENAME)
                print(f"Model extracted and loaded from {MODEL_FILENAME}.")
                return True
            else:
                print(f"Extracted model file {MODEL_FILENAME} not found.")
                return False
        except zipfile.BadZipFile:
            print("Invalid ZIP file.")
            return False
    else:
        print(f"Model ZIP file {MODEL_ZIP} not found.")
    
    return False




def download_supporting_files():
    """
    Download additional supporting files for the model.
    """
    try:
        # Define the URLs for target encodings and low importance features
        target_encodings_url = "https://storage.googleapis.com/fast_api_crop_production/target_encodings.pkl"
        low_importance_features_url = "https://storage.googleapis.com/fast_api_crop_production/low_importance_features.pkl"
        
        # Download target encodings
        response_encodings = requests.get(target_encodings_url)
        response_encodings.raise_for_status()
        with open('target_encodings.pkl', 'wb') as encodings_file:
            encodings_file.write(response_encodings.content)
        
        # Download low importance features
        response_features = requests.get(low_importance_features_url)
        response_features.raise_for_status()
        with open('low_importance_features.pkl', 'wb') as features_file:
            features_file.write(response_features.content)
        
    except Exception as e:
        print(f"Error downloading supporting files: {e}")
        raise






# app = FastAPI(
#     title="Crop Production Prediction API",
#     description="API for predicting crop production using a Random Forest model",
#     version="1.0.0"
# )
try:
    download_supporting_files()
    if download_model_from_gcs():
         extract_and_load_model()
except Exception as e:
    model = None
    print(f"Failed to download model or supporting files: {e}")
    raise

# Download model and supporting files when the application starts
# try:
#     model_path = download_model_from_gcs()
#     download_supporting_files()
# except Exception as e:
#     print(f"Failed to download model or supporting files: {e}")
#     # You might want to handle this more gracefully in a production environment
#     raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class CropInput(BaseModel):
    Area: float = Field(..., gt=0, description="Area under cultivation in hectares")
    Crop_Year: int = Field(..., ge=1996, le=2030, description="Year of cultivation")
    State_Name: str = Field(..., min_length=2, description="Name of the state")
    District_Name: str = Field(..., min_length=2, description="Name of the district")
    Season: str = Field(..., description="Growing season (e.g., Kharif, Rabi)")
    Crop: str = Field(..., description="Type of crop")

    # Validator for Area
    @field_validator('Area')
    @classmethod
    def area_must_be_reasonable(cls, v):
        if v > 8580100:  # Maximum reasonable area in hectares
            raise ValueError('Area is too large. Maximum allowed is 8,580,100 hectares')
        return v
    
    # Validator for Crop_Year
    @field_validator('Crop_Year')
    @classmethod
    def year_must_be_reasonable(cls, v):
        current_year = 2025
        if v > 2035:
            raise ValueError(f'Crop year cannot be predicted beyond 2035. Too far for data trends to be reliable. Current year is {current_year}')
        return v
    
    # Validator for State_Name
    @field_validator('State_Name')
    @classmethod
    def state_name_must_be_valid(cls, v):
        valid_states = {
            'Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 
            'Chhattisgarh', 'Dadra and Nagar Haveli', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 
            'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 
            'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 
            'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
        }
        
        v = v.strip().title()  # Normalize input

        if v not in valid_states:
            raise ValueError(f"'{v}' is not a valid Indian state or union territory")

        return v
    
    # Validator for District_Name
    @field_validator('District_Name')
    @classmethod
    def district_name_must_be_valid(cls, v):
        v = v.strip().upper()  # Normalize input
        if not v.replace(" ", "").isalpha():
            raise ValueError('District name must contain only alphabetic characters')

        # Keep valid district set externally for maintainability

        valid_districts = [
            'NICOBARS', 'NORTH AND MIDDLE ANDAMAN', 'SOUTH ANDAMANS', 'ANANTAPUR',
            'CHITTOOR', 'EAST GODAVARI', 'GUNTUR', 'KADAPA', 'KRISHNA', 'KURNOOL',
            'PRAKASAM', 'SPSR NELLORE', 'SRIKAKULAM', 'VISAKHAPATNAM', 'VIZIANAGARAM',
            'WEST GODAVARI', 'ANJAW', 'CHANGLANG', 'DIBANG VALLEY', 'EAST KAMENG',
            'EAST SIANG', 'KURUNG KUMEY', 'LOHIT', 'LONGDING', 'LOWER DIBANG VALLEY',
            'LOWER SUBANSIRI', 'NAMSAI', 'PAPUM PARE', 'TAWANG', 'TIRAP', 'UPPER SIANG',
            'UPPER SUBANSIRI', 'WEST KAMENG', 'WEST SIANG', 'BAKSA', 'BARPETA',
            'BONGAIGAON', 'CACHAR', 'CHIRANG', 'DARRANG', 'DHEMAJI', 'DHUBRI',
            'DIBRUGARH', 'DIMA HASAO', 'GOALPARA', 'GOLAGHAT', 'HAILAKANDI', 'JORHAT',
            'KAMRUP', 'KAMRUP METRO', 'KARBI ANGLONG', 'KARIMGANJ', 'KOKRAJHAR',
            'LAKHIMPUR', 'MARIGAON', 'NAGAON', 'NALBARI', 'SIVASAGAR', 'SONITPUR',
            'TINSUKIA', 'UDALGURI', 'ARARIA', 'ARWAL', 'AURANGABAD', 'BANKA',
            'BEGUSARAI', 'BHAGALPUR', 'BHOJPUR', 'BUXAR', 'DARBHANGA', 'GAYA',
            'GOPALGANJ', 'JAMUI', 'JEHANABAD', 'KAIMUR (BHABUA)', 'KATIHAR',
            'KHAGARIA', 'KISHANGANJ', 'LAKHISARAI', 'MADHEPURA', 'MADHUBANI', 'MUNGER',
            'MUZAFFARPUR', 'NALANDA', 'NAWADA', 'PASHCHIM CHAMPARAN', 'PATNA',
            'PURBI CHAMPARAN', 'PURNIA', 'ROHTAS', 'SAHARSA', 'SAMASTIPUR', 'SARAN',
            'SHEIKHPURA', 'SHEOHAR', 'SITAMARHI', 'SIWAN', 'SUPAUL', 'VAISHALI',
            'CHANDIGARH', 'BALOD', 'BALODA BAZAR', 'BALRAMPUR', 'BASTAR', 'BEMETARA',
            'BIJAPUR', 'BILASPUR', 'DANTEWADA', 'DHAMTARI', 'DURG', 'GARIYABAND',
            'JANJGIR-CHAMPA', 'JASHPUR', 'KABIRDHAM', 'KANKER', 'KONDAGAON', 'KORBA',
            'KOREA', 'MAHASAMUND', 'MUNGELI', 'NARAYANPUR', 'RAIGARH', 'RAIPUR',
            'RAJNANDGAON', 'SUKMA', 'SURAJPUR', 'SURGUJA', 'DADRA AND NAGAR HAVELI',
            'NORTH GOA', 'SOUTH GOA', 'AHMADABAD', 'AMRELI', 'ANAND', 'BANAS KANTHA',
            'BHARUCH', 'BHAVNAGAR', 'DANG', 'DOHAD', 'GANDHINAGAR', 'JAMNAGAR',
            'JUNAGADH', 'KACHCHH', 'KHEDA', 'MAHESANA', 'NARMADA', 'NAVSARI',
            'PANCH MAHALS', 'PATAN', 'PORBANDAR', 'RAJKOT', 'SABAR KANTHA', 'SURAT',
            'SURENDRANAGAR', 'TAPI', 'VADODARA', 'VALSAD', 'AMBALA', 'BHIWANI',
            'FARIDABAD', 'FATEHABAD', 'GURGAON', 'HISAR', 'JHAJJAR', 'JIND', 'KAITHAL',
            'KARNAL', 'KURUKSHETRA', 'MAHENDRAGARH', 'MEWAT', 'PALWAL', 'PANCHKULA',
            'PANIPAT', 'REWARI', 'ROHTAK', 'SIRSA', 'SONIPAT', 'YAMUNANAGAR',
            'CHAMBA', 'HAMIRPUR', 'KANGRA', 'KINNAUR', 'KULLU', 'LAHUL AND SPITI',
            'MANDI', 'SHIMLA', 'SIRMAUR', 'SOLAN', 'UNA', 'ANANTNAG', 'BADGAM',
            'BANDIPORA', 'BARAMULLA', 'DODA', 'GANDERBAL', 'JAMMU', 'KARGIL', 'KATHUA',
            'KISHTWAR', 'KULGAM', 'KUPWARA', 'LEH LADAKH', 'POONCH', 'PULWAMA',
            'RAJAURI', 'RAMBAN', 'REASI', 'SAMBA', 'SHOPIAN', 'SRINAGAR', 'UDHAMPUR'
        ]

        if v not in valid_districts:
            raise ValueError('District name must be a valid Indian district')

        return v

    # Validator for Season
    @field_validator('Season')
    @classmethod
    def season_must_be_valid(cls, v):
        valid_seasons = {'Kharif', 'Whole year', 'Autumn', 'Rabi', 'Summer', 'Winter'}
        v = v.strip().capitalize()  # Normalize input

        if v not in valid_seasons:
            raise ValueError(f"'{v}' is not a valid season. Must be one of {valid_seasons}")

        return v

    # Validator for Crop
    @field_validator('Crop')
    @classmethod
    def crop_must_be_valid(cls, v):

        # Define valid crops (case-insensitive lookup)
        valid_crops = [
            'Arecanut', 'Other Kharif pulses', 'Rice', 'Banana', 'Cashewnut', 'Coconut',
            'Dry ginger', 'Sugarcane', 'Sweet potato', 'Tapioca', 'Black pepper',
            'Dry chillies', 'other oilseeds', 'Turmeric', 'Maize', 'Moong(Green Gram)',
            'Urad', 'Arhar/Tur', 'Groundnut', 'Sunflower', 'Bajra', 'Castor seed',
            'Cotton(lint)', 'Horse-gram', 'Jowar', 'Korra', 'Ragi', 'Tobacco', 'Gram',
            'Wheat', 'Masoor', 'Sesamum', 'Linseed', 'Safflower', 'Onion',
            'other misc. pulses', 'Samai', 'Small millets', 'Coriander', 'Potato',
            'Other Rabi pulses', 'Soyabean', 'Beans & Mutter(Vegetable)', 'Bhindi',
            'Brinjal', 'Citrus Fruit', 'Cucumber', 'Grapes', 'Mango', 'Orange',
            'other fibres', 'Other Fresh Fruits', 'Other Vegetables', 'Papaya',
            'Pome Fruit', 'Tomato', 'Rapeseed &Mustard', 'Mesta', 'Cowpea(Lobia)', 'Lemon',
            'Pome Granet', 'Sapota', 'Cabbage', 'Peas (vegetable)', 'Niger seed',
            'Bottle Gourd', 'Sannhamp', 'Varagu', 'Garlic', 'Ginger', 'Oilseeds total',
            'Pulses total', 'Jute', 'Peas & beans (Pulses)', 'Blackgram', 'Paddy',
            'Pineapple', 'Barley', 'Khesari', 'Guar seed', 'Moth',
            'Other Cereals & Millets', 'Cond-spcs other', 'Turnip', 'Carrot', 'Redish',
            'Arcanut (Processed)', 'Atcanut (Raw)', 'Cashewnut Processed',
            'Cashewnut Raw', 'Cardamom', 'Rubber', 'Bitter Gourd', 'Drum Stick',
            'Jack Fruit', 'Snak Guard', 'Pump Kin', 'Tea', 'Coffee', 'Cauliflower',
            'Other Citrus Fruit', 'Water Melon', 'Total foodgrain', 'Kapas', 'Colocosia',
            'Lentil', 'Bean', 'Jobster', 'Perilla', 'Rajmash Kholar', 'Ricebean (nagadal)',
            'Ash Gourd', 'Beet Root', 'Lab-Lab', 'Ribed Guard', 'Yam', 'Apple', 'Peach',
            'Pear', 'Plums', 'Litchi', 'Ber', 'Other Dry Fruit', 'Jute & mesta'
        ]

        v = v.strip().title()  # Normalize input

        if v not in valid_crops:
            raise ValueError(f"'{v}' is not a recognized crop")

        return v


    class Config:
        schema_extra = {
            "example": {
                "Area": 100.5,
                "Crop_Year": 2023,
                "State_Name": "Maharashtra",
                "District_Name": "Nicobars",
                "Season": "Kharif",
                "Crop": "Rice"
            }
        } 


class PredictionResponse(BaseModel):
    predicted_production: float
    input_data: CropInput


# Function to predict crop production
def predict_crop_production(input_data, model_path='bestmodel.pkl',
                        target_encodings_path='target_encodings.pkl',
                        low_importance_features_path='low_importance_features.pkl'):
    """
    Predicts crop production using the trained Random Forest model.
    
    Parameters:
    - input_data (dict): A dictionary containing the input features:
        - 'Area': Area under cultivation
        - 'Crop_Year': Year of cultivation
        - 'State_Name': Name of the state
        - 'District_Name': Name of the district
        - 'Season': Growing season
        - 'Crop': Type of crop
    - model_path (str): Path to the saved model
    - target_encodings_path (str): Path to the saved target encodings
    - low_importance_features_path (str): Path to the low-importance features
    
    Returns:
    - float: Predicted crop production in tons
    """
    import numpy as np
    import pandas as pd
    import joblib
    
    # Load all saved objects
    model = joblib.load(model_path)
    target_encodings = joblib.load(target_encodings_path)
    low_importance_features = joblib.load(low_importance_features_path)
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Step 1: Apply log transform to Area (same as in training)
    input_df['Area'] = np.log1p(input_df['Area'])
    
    # Step 2: Apply target encoding to categorical columns
    categorical_cols = ["State_Name", "District_Name", "Season", "Crop"]
    
    # Create a copy for safe manipulation
    input_df_copy = input_df.copy()
    
    for col in categorical_cols:
        if col in input_df_copy.columns:
            # Map the categorical values using the same encoding as in training
            # target_encodings[col] is a Series with index=category and values=encoding
            input_df_copy[col] = input_df_copy[col].map(lambda x: target_encodings[col].get(x) if x in target_encodings[col].index else None)
    
    # Handle unseen categories (NaN values) using the mean of training encodings
    for col in categorical_cols:
        if col in input_df_copy.columns and input_df_copy[col].isnull().any():
            # Calculate mean of the target encodings for this column
            mean_encoding = target_encodings[col].mean()
            input_df_copy[col] = input_df_copy[col].fillna(mean_encoding)
    
    # Step 3: Ensure all required columns exist
    required_features = model.feature_names_in_
    
    # Add any missing columns that the model expects
    for col in required_features:
        if col not in input_df_copy.columns:
            input_df_copy[col] = 0  # Use a default value
    
    # Step 4: Drop low importance features
    if low_importance_features is not None:
        for col in low_importance_features:
            if col in input_df_copy.columns:
                input_df_copy = input_df_copy.drop(columns=[col])
    
    # Step 5: Make sure columns are in the exact same order as training
    input_df_copy = input_df_copy[required_features]
    
    # Step 6: Make prediction
    prediction_log = model.predict(input_df_copy)
    
    # Step 7: Convert back from log scale to original scale
    prediction = np.expm1(prediction_log[0])
    
    return prediction


# Create prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: CropInput):
    
    try:
        # Convert Pydantic model to dictionary
        input_dict = input_data.model_dump()
        print("Received Input Data:", input_dict)
        
        # Get prediction
        prediction = predict_crop_production(
            input_dict, 
            model_path='best_model.pkl',
            target_encodings_path='target_encodings.pkl',
            low_importance_features_path='low_importance_features.pkl'
        )

        # if model is None:
        #         raise HTTPException(status_code=500, detail="Model is not loaded. Try reloading.")


        
        # Return prediction and input data
        return {
            "predicted_production": f"{float(prediction):.2f}",  # Ensures exactly 2 decimal places
            "input_data": input_data
}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") 
    


@app.get("/reload/")
async def reload_model():
    """
    Reloads the model by downloading and extracting it again.
    """
    if download_model_from_gcs() and extract_and_load_model():
        return {"message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")
    


@app.get("/")
async def root():
    return {"message": "Crop Production Prediction API is running. Go to /docs for the API documentation."}


if __name__ == "__main__":
    # Use the PORT environment variable provided by the hosting platform, or default to 8000
    print("Starting server...")
    port = int(os.environ.get("PORT", 8000))
    print(f"Attempting to run on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)

