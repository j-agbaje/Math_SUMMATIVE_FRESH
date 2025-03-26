from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, Field, field_validator
import joblib
import os
import uvicorn
import asyncio
from functools import lru_cache
import requests
import zipfile
from sklearn.tree import DecisionTreeClassifier
from google.cloud import storage
import numpy as np
import pandas as pd
import gcsfs
import json
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




@lru_cache(maxsize=1)
def load_gcs_credentials():
    """Safely load and parse GCS credentials."""
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not credentials_json:
        raise ValueError("GCS credentials not found. Ensure GOOGLE_APPLICATION_CREDENTIALS_JSON is set.")
    return json.loads(credentials_json)

async def load_model_from_gcs(max_retries=3):
    """
    Load model from Google Cloud Storage with retry mechanism.
    
    Args:
        max_retries (int): Number of times to retry loading the model
    
    Returns:
        Loaded model object
    """
    global global_model
    
    for attempt in range(max_retries):
        try:
            # Get credentials
            credentials_dict = load_gcs_credentials()
            
            # Create GCS Filesystem
            fs = gcsfs.GCSFileSystem(token=credentials_dict)
            
            # Path to the model in GCS
            model_path = "gs://fast_api_crop_production/best_model.pkl"
            
            # Load model directly from GCS
            with fs.open(model_path, 'rb') as f:
                global_model = joblib.load(f)
            
            print("Model loaded successfully from Google Cloud Storage.")
            return global_model
        
        except Exception as e:
            print(f"Attempt {attempt + 1} failed. Error: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to load model after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def load_supporting_files_from_gcs(max_retries=3):
    """
    Load supporting files from Google Cloud Storage with retry mechanism.
    
    Args:
        max_retries (int): Number of times to retry loading the files
    
    Returns:
        Tuple of (target_encodings, low_importance_features)
    """
    global global_target_encodings, global_low_importance_features
    
    for attempt in range(max_retries):
        try:
            # Get credentials
            credentials_dict = load_gcs_credentials()
            
            # Create GCS Filesystem
            fs = gcsfs.GCSFileSystem(token=credentials_dict)
            
            # Paths to supporting files in GCS
            target_encodings_path = "gs://fast_api_crop_production/target_encodings.pkl"
            low_importance_features_path = "gs://fast_api_crop_production/low_importance_features.pkl"
            
            # Load target encodings
            with fs.open(target_encodings_path, 'rb') as f:
                global_target_encodings = joblib.load(f)
            
            # Load low importance features
            with fs.open(low_importance_features_path, 'rb') as f:
                global_low_importance_features = joblib.load(f)
            
            print("Supporting files loaded successfully from Google Cloud Storage.")
            return global_target_encodings, global_low_importance_features
        
        except Exception as e:
            print(f"Attempt {attempt + 1} failed. Error: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to load supporting files after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(2 ** attempt)
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


    def to_dict(self, **kwargs):
        return super().model_dump(**kwargs)

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
def predict_crop_production(
    input_data, 
    model, 
    target_encodings, 
    low_importance_features
):
    """
    Predicts crop production using the trained Random Forest model.
    
    Parameters:
    - input_data (dict): A dictionary containing the input features
    - model: Loaded Random Forest model
    - target_encodings: Dictionary of target encodings for categorical features
    - low_importance_features: List of features to drop
    
    Returns:
    - float: Predicted crop production in tons
    """
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
    global global_model, global_target_encodings, global_low_importance_features
    
    # Load model and supporting files if not already loaded
    if global_model is None or global_target_encodings is None or global_low_importance_features is None:
        try:
            # Load model and supporting files concurrently
            global_model, (global_target_encodings, global_low_importance_features) = await asyncio.gather(
                load_model_from_gcs(),
                load_supporting_files_from_gcs()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model or supporting files: {str(e)}")
    
    try:
        # Convert Pydantic model to dictionary
        input_dict = input_data.to_dict()
        print("Received Input Data:", input_dict)
        
        # Get prediction
        prediction = predict_crop_production(
            input_dict,
            model=global_model,
            target_encodings=global_target_encodings,
            low_importance_features=global_low_importance_features
        )
        
        # Return prediction and input data
        return {
            "predicted_production": f"{float(prediction):.2f}",  # Ensures exactly 2 decimal places
            "input_data": input_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    


# @app.get("/reload/")
# async def reload_model():
#     """
#     Reloads the model by downloading and extracting it again.
#     """
#     if download_model_from_gcs() and extract_and_load_model():
#         return {"message": "Model reloaded successfully"}quire
#     else:
#         raise HTTPException(status_code=500, detail="Failed to reload model")
    


@app.get("/")
async def root():
    return {"message": "Crop Production Prediction API is running. Go to /docs for the API documentation."}


if __name__ == "__main__":
    # Use the PORT environment variable provided by the hosting platform, or default to 8000
    print("Starting server...")
    port = int(os.environ.get("PORT", 8000))
    print(f"Attempting to run on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)

