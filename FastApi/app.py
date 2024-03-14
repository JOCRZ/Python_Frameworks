
# 1. Library imports
import uvicorn
from fastapi import FastAPI
from input import inputdata
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()
pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

# 3. Define API endpoint
@app.post('/predict')
def predict(data: inputdata):
    data = data.dict()
    experience = data['experience']
    testscore = data['testscore']
    interviewscore = data['interviewscore']
    prediction = model.predict([[experience, testscore, interviewscore]])[0]  # Access the first prediction
    return {"prediction": prediction}

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

    
#uvicorn app:app --reload