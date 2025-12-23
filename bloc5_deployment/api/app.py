import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from typing import Literal, List, Union
import mlflow

#--------------------------
# Configuration
#--------------------------

description = """
Welcome to the **Get Around price prediction API**!

This API predicts the **optimal daily rental price** based on the characteristics of a car.

The following endpoints are available:
* `/` (GET): returns a welcome message.
* `/docs`: displays the API documentation.
* `/predict` (POST): you can provide your car characteristics and receive the prediction for the daily rental price.

"""

tag_metadata = [
    {
        "name": "Prediction",
        "description": "Prediction of the car daily rental price"
    }
]

app = FastAPI(
    title="Get Around price prediction API",
    description=description,
    openapi_tags= tag_metadata
)

class PredictionFeatures(BaseModel):
    model_key: Literal['Alfa Romeo', 'Audi', 'BMW', 'Citroën', 'Ferrari', 'Fiat', 
                    'Ford', 'Honda', 'KIA Motors', 'Lamborghini', 'Lexus', 'Mazerati', 
                    'Mazda', 'Mercedes', 'Mini', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 
                    'PGO', 'Porsche', 'Renault', 'SEAT', 'Subaru', 'Suzuki', 
                    'Toyota', 'Volkswagen', 'Yamaha']
    mileage: Union[int, float] = Field(
        gt=0,
        description="Mileage of the car in kilometers (must be strictly positive)"
    )
    engine_power: int = Field(
        ge=85,
        le=240,
        description="Engine power in horsepower (between 85 and 240)"
    )
    fuel: Literal['diesel', 'electro', 'hybrid_petrol', 'petrol']
    paint_color: Literal['beige', 'black', 'blue', 'brown', 'green', 'grey', 'orange', 'red', 'silver', 'white']
    car_type: Literal['convertible', 'coupe', 'estate', 'hatchback', 'sedan','subcompact', 'suv', 'van']
    private_parking_available: bool = Field(..., description="True if private parking is available")
    has_gps: bool = Field(..., description="True if the car has GPS")
    has_air_conditioning: bool = Field(..., description="True if the car has air conditioning")
    automatic_car: bool = Field(..., description="True if the car has automatic transmission")
    has_getaround_connect: bool = Field(..., description="True if the car has Getaround Connect")
    has_speed_regulator: bool = Field(..., description="True if the car has a speed regulator control")
    winter_tires: bool = Field(..., description="True if the car has winter tires")


#--------------------------
# Define endpoints
#--------------------------

@app.get("/")
async def index():
    message = "Welcome to the Get Around price prediction API. If you want to learn more, check out the documentation of the API at `/docs`"
    return message
    
@app.post("/predict", tags=["Prediction"])
async def predict(features: PredictionFeatures):
    data = pd.DataFrame({"model_key":[features.model_key],
                        "mileage":[features.mileage],
                        "engine_power":[features.engine_power],
                        "fuel":[features.fuel],
                        "paint_color":[features.paint_color],
                        "car_type":[features.car_type],
                        "private_parking_available":[features.private_parking_available],
                        "has_gps":[features.has_gps],
                        "has_air_conditioning":[features.has_air_conditioning],
                        "automatic_car":[features.automatic_car],
                        "has_getaround_connect":[features.has_getaround_connect],
                        "has_speed_regulator":[features.has_speed_regulator],
                        "winter_tires":[features.winter_tires],
                        })
    
    print(features)
    
    logged_model = 'runs:/fd4245cbd4e84a03b0032a23d10437f8/get_around_pricing_model'

    # Load model as a PyFuncModel
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame
    prediction = loaded_model.predict(data)

    response = {
        "prediction": round(prediction.tolist()[0], 1)
    }
    
    return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)