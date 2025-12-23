# Get Around project

GetAround is a peer-to-peer car rental platform, similar to Airbnb for vehicles.

This project aims to address two main issues:

- **Management of late returns**: avoid conflicts between consecutive bookings by implementing a minimum delay between two rentals.
- **Price optimization**: develop a machine learning model to suggest optimal prices to car owners.


## Repository structure
```text
.
├── data/               # Raw datasets used for analysis and training
│   ├── get_around_delay_analysis.xlsx
│   └── get_around_pricing_project.csv
├── notebooks/          # Exploration and analysis notebooks
│   ├── get_around_dashboard.ipynb
│   └── get_around_pricing.ipynb
├── training/           # Model training and MLflow experiments
│   ├── train.py
│   ├── run.bat         # Windows script to run the Docker container
│   ├── Dockerfile
│   └── requirements.txt
├── models/             # Trained model artifacts
│   ├── model_lasso.pkl
│   ├── model_ridge.pkl
│   ├── model_rf.pkl
│   └── model_xgb.pkl
├── api/                # FastAPI application for price prediction
│   ├── api.py
│   ├── run.bat         # Windows script to run the Docker container
│   ├── Dockerfile
│   ├── requirements.txt
│   └── tests/          # API tests (curl & requests)
├── dashboard/          # Streamlit dashboard
│   ├── .streamlit/     # Streamlit configuration files
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── .gitignore
└── README.md
```

## Project workflow

1. Data exploration and analysis in `notebooks/`
2. Business insights available via a Streamlit dashboard in `dashboard/`
3. Model training and experiment tracking using MLflow in `training/`
4. Model serving through a FastAPI application in `api/`


## Deliverables

* [Dashboard](https://huggingface.co/spaces/csil08/get_around_dashboard) to explore the impact of late returns and simulate a minimum buffer time between bookings    
* [Predict endpoint](https://your-username-getaround-api.hf.space/predict) returning price suggestions from a trained machine learning model
* [API Documentation](https://huggingface.co/spaces/your-username/getaround-api) 
* [MLflow Experiment Tracking](https://csil08-mlflow-server-demo.hf.space/)


## Tech Stack

| **Component**       | **Technologies**                          |
|---------------------|------------------------------------------|
| Dashboard           | Streamlit, Plotly                        |
| API                 | FastAPI, Uvicorn                         |
| Machine learning    | MLflow, Scikit-learn                     |
| Containerization    | Docker                                   |
| Hosting             | Hugging Face Spaces                      |

## Machine learning training

The `training/` folder contains the model training pipeline and experiment tracking using **MLflow**.

Four models are trained: 
- two regularized linear regression models: **Ridge** and **Lasso**;
- **Random Forest**;
- **XGBoost**.

The `train.py` script allows the user which model to train using the `--model` flag. By default, XGBoost (the preferred model) is selected.

Model artifacts and metrics are logged to the MLflow server hosted on Hugging Face. In addition, trained models are also saved locally in the `models/` folder for convenience.

**Important:** 
- While it is possible to run MLflow locally, this project uses a remote **Hugging Face-hosted server**. An AWS S3 bucket was used to store artifacts and a PostgreSQL database (via Neon) was used as the backend store.   
- To run the training or the API with your own models, you must set up your own MLflow server and configure any artifact storage and backend database as needed.


## API

The `api/` folder provides a **FastAPI** application:
- Input validation with **Pydantic**
- Prediction endpoint: `/predict`
- Interactive documentation accessible via `/docs`
- Tests using `curl` and Python `requests`

The API retrieves the XGBoost trained model logged in MLflow in order to make the predictions. 

**Important:** 
- The code is currently configured with the URI of a specific MLflow run on a Hugging Face Spaces server. 
- To run the API, you must set up your own MLflow server and update the model URI in the `api.py` script.  


### Run API locally

```bash
# Navigate to the API folder
cd api

# Build the Docker image
docker build . -t api-image

# Run the API container (Windows)
.\run.bat
```

### Access the API locally  
- Open API docs: http://localhost:4000/docs   
- Predict endpoint: http://localhost:4000/predict


## Dashboard

The `dashboard/` folder contains a Streamlit app which:
- displays car rentals and late returns statistics
- simulates scenarios of minimum buffer time

### Run dashboard locally

```bash
# Navigate to the dashboard folder
cd dashboard

# Build the Docker image
docker build . -t dashboard-image

# Run the dashboard container
docker run -p 8501:8501 dashboard-image
```
### Access the dashboard locally
Navigate to http://localhost:8501 in your browser