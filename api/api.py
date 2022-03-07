from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ast
import random
import tensorflow as tf
from tensorflow import keras

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get('/')
def greeting():
    return {'greeting':'you ding a big man ting bruv.'}

@app.get('/predict')
def note(sequence):
    #-----grabbing the model-----
    model = keras.models.load_model("model/model.keras")

    #-----transform the sequence to the format model can take in-----
    #sequence example : [actual pitch, actual duration]
    input_sequence = ast.literal_eval(sequence)

    #-----take in the sequence-----

    prediction = model.predict(input_sequence)

    #-----transform the prediction into the notes the front-end can take in-----
    #----> grab it from stephen

    return {'predictions': }
