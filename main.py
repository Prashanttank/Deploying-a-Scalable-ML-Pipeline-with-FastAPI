import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# Define the Pydantic data model
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Load the model and encoder
project_path = "/workspace/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
model_path = os.path.join(project_path, "model", "model.pkl")
encoder = load_model(encoder_path)
model = load_model(model_path)

# Create FastAPI app instance
app = FastAPI()

# Define the root endpoint
@app.get("/")
async def get_root():
    """Say hello!"""
    return {"message": "Welcome to the Income Prediction API!"}

# Define the inference endpoint
@app.post("/predict/")
async def post_inference(data: Data):
    # Turn the Pydantic model into a dictionary
    data_dict = data.dict()
    print("Received data:", data_dict)  # Debug print
    
    # Clean up the dict to turn it into a Pandas DataFrame
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)
    print("Data after formatting:", data)  # Debug print

    # Process data using the process_data function
    try:
        data_processed, _, _, _ = process_data(
            data,
            categorical_features=[
                "workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country",
            ],
            label=None,
            training=False,
            encoder=encoder
        )
        print("Processed data:", data_processed)  # Debug print
    except Exception as e:
        print("Error during data processing:", str(e))  # Catch any data processing errors
        return {"error": "Data processing error", "details": str(e)}

    # Ensure data_processed is a 2D array for inference
    if data_processed.ndim == 1:
        data_processed = data_processed.reshape(1, -1)
    
    # Make inference using the trained model
    try:
        _inference = inference(model, data_processed)
        if isinstance(_inference, (list, pd.Series, pd.DataFrame)):
            result = apply_label(_inference[0])
        else:
            result = apply_label(_inference)  # Handle case where _inference is scalar
        print("Inference result:", result)  # Debug print
    except Exception as e:
        print("Error during inference:", str(e))  # Catch any inference errors
        return {"error": "Internal Server Error", "details": str(e)}

    return {"result": result}
