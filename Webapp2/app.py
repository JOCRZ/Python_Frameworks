# 1. Library imports
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from input import inputdata
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # Serve static files

pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    with open("static/index.html") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)

# Define API endpoint
@app.post('/predict')
def predict(data: inputdata):
    data = data.dict()
    experience = data['experience']
    testscore = data['testscore']
    interviewscore = data['interviewscore']
    prediction = model.predict([[experience, testscore, interviewscore]])[0]  # Access the first prediction
    return {"prediction": prediction}

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
