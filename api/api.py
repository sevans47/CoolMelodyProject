from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ast
import random

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
    li = ast.literal_eval(sequence)
    i = random.randint(0,4)
    next_note = li[i]
    return {"notes" : str(next_note)}
