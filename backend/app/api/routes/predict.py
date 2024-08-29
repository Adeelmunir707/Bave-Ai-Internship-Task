from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import pickle
from fastapi import APIRouter, HTTPException

# Define the FastAPI app
router = APIRouter()

# Load the TF-IDF vectorizer and SVM model from pickled files
vectorizer = joblib.load("./vectorizer.pkl")
model = joblib.load("./model.pkl")

# Define the input data model
class EmailContent(BaseModel):
    text: str

# Define the prediction endpoint
@router.post("/predict/")
def predict_spam(email: EmailContent):
    
    # Vectorize the input text using the loaded TF-IDF vectorizer
    vectorized_text = vectorizer.transform([email.text])

    # Predict the label using the loaded SVM model
    prediction = model.predict(vectorized_text)

    # Convert the prediction to human-readable format
    prediction_label = "Spam" if prediction == 1 else "Not Spam"

    return {"prediction": prediction_label}


