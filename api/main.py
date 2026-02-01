from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_price, batch_predict
from house_price_predictor.config.schemas import HousePredictionRequest, PredictionResponse
# Initialize FastAPI app with metadata
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "An API for predicting house prices based on various features. "
        "This application is part of the MLOps Bootcamp by School of Devops course. "
        "Authored by Gourav Shah and modified by Wesley Wood"
    ),
    version="1.0.0",
    contact={
        "name": "School of Devops",
        "url": "https://schoolofdevops.com",
        "email1": "learn@schoolofdevops.com",
        "email2": "wwood@ab.bluecross.ca"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=t.Dict)
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    return predict_price(request)

# Batch prediction endpoint
@app.post("/batch-predict", response_model=list(PredictionResponse))
async def batch_predict_endpoint(requests: list[HousePredictionRequest]):
    return batch_predict(requests)