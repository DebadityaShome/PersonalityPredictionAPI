import uvicorn
from fastapi import FastAPI
from Personality import tweet
import numpy as np 
import pickle 
import pandas as pd 

app = FastAPI()
loaded_model = pickle.load(open('big5.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pickle', 'rb'))

@app.get('/')
def index():
    return {'message': 'Pass in any text to get personality prediction in terms of Big5 scores'}

@app.post('/predict')
def predict_personality(data:tweet):
    data = data.dict()
    text = str(data['tweets'])

    result = loaded_model.predict(tfidf.transform([text]))[0]

    return {
        'Extraversion' : result[0],
        'Neuroticism': result[1],
        'Agreeableness': result[2],
        'Conscientiousness': result[3],
        'Openness': result[4]
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)