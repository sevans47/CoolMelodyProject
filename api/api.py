from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ast

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


@app.get('/predict')
def note(sequence):
    li = ast.literal_eval(sequence)
    next_note = li[0][0]
    return {"notes" : str(next_note)}
