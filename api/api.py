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

pitch_symb = ['0',
 '49',
 '50',
 '51',
 '52',
 '53',
 '54',
 '55',
 '56',
 '57',
 '58',
 '59',
 '60',
 '61',
 '62',
 '63',
 '64',
 '65',
 '66',
 '67',
 '68',
 '69',
 '70',
 '71',
 '72',
 '73',
 '74',
 '75',
 '76',
 '77',
 '78',
 '79',
 '80',
 '81',
 '82',
 '83',
 '84',
 '85',
 '86',
 '87',
 '88',
 '89',
 '90',
 '91',
 '92']
duration_symb = ['1', '16', '16d', '2', '2d', '32', '32d', '4', '4d', '64', '8', '8d', '8dd']

L_pitch_symb = len(pitch_symb)
L_duration_symb = len(duration_symb)

len_in_64th_notes = {'64': 1, '32': 2, '32d': 3, '16': 4, '16d': 6, '8': 8, '8d': 12,
                     '8dd': 14, '4': 16, '4d': 24, '2': 32, '2d': 48, '1': 64}
duration_mapping = {1: 9,
 2: 5,
 3: 6,
 4: 1,
 6: 2,
 8: 10,
 12: 11,
 14: 12,
 16: 7,
 24: 8,
 32: 3,
 48: 4,
 64: 0}
pitch_mapping = {0: 0,
 49: 1,
 50: 2,
 51: 3,
 52: 4,
 53: 5,
 54: 6,
 55: 7,
 56: 8,
 57: 9,
 58: 10,
 59: 11,
 60: 12,
 61: 13,
 62: 14,
 63: 15,
 64: 16,
 65: 17,
 66: 18,
 67: 19,
 68: 20,
 69: 21,
 70: 22,
 71: 23,
 72: 24,
 73: 25,
 74: 26,
 75: 27,
 76: 28,
 77: 29,
 78: 30,
 79: 31,
 80: 32,
 81: 33,
 82: 34,
 83: 35,
 84: 36,
 85: 37,
 86: 38,
 87: 39,
 88: 40,
 89: 41,
 90: 42,
 91: 43,
 92: 44}

@app.get('/')
def greeting():
    return {'greeting':'you ding a big man ting bruv.'}

@app.get('/predict')
def note(sequence):
    #-----grabbing the model-----
    model = keras.models.load_model("model/model.keras")

    #-----transform the sequence to the format model can take in-----
    #sequence example : [actual pitch, actual duration]
    list_sequence = ast.literal_eval(sequence)
    input_sequence = []

    for note in list_sequence:
        dur_len_in_64th_notes = len_in_64th_notes[note[1]]
        dur_mapped = duration_mapping[dur_len_in_64th_notes]
        pitch_mapped = pitch_mapping[note[0]]
        mapped_note = [pitch_mapped, dur_mapped]
        note_normalized = [mapped_note[0]/float(L_pitch_symb), mapped_note[1]/float(L_duration_symb)]
        input_sequence.append(note_normalized)

    #-----take in the sequence-----

    prediction = model.predict(input_sequence)

    #-----transform the prediction into the notes the front-end can take in-----
    #----> grab it from stephen

    return {'predictions': }
