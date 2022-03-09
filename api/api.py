from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ast
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


pitch_symb = [ 0, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
       45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
       62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
       79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96]

duration_symb = [0.    , 0.0625, 0.125 , 0.25  , 0.375 , 0.5   , 0.75  , 1.    ,
       1.25  , 1.5   , 1.75  , 2.    , 2.25  , 2.5   , 2.75  , 3.    ,
       3.25  , 3.5   , 3.75  , 4.    , 4.5   , 5.    , 6.    , 7.5   ,
       8.75  , 9.5   ]

L_pitch_symb = len(pitch_symb)
L_duration_symb = len(duration_symb)

len_in_64th_notes = {'64': 1, '32': 2, '32d': 3, '16': 4, '16d': 6, '8': 8, '8d': 12,
                     '8dd': 14, '4': 16, '4d': 24, '2': 32, '2d': 48, '1': 64}


reverse_len_in_64th_notes = {v: k for k, v in len_in_64th_notes.items()}


pitch_mapping = dict((int(c), i) for i, c in enumerate(pitch_symb))
pitch_reverse_mapping = dict((i, int(c)) for i, c in enumerate(pitch_symb))
duration_mapping = dict((float(c), i) for i, c in enumerate(duration_symb))
duration_reverse_mapping = dict((i, float(c)) for i, c in enumerate(duration_symb))

@app.get('/')
def greeting():
    return {'greeting':'you ding a big man ting bruv.'}

@app.get('/initialize')
def first_sequence():
    #-----generate data to randomly grab data from-----
    random_value = random.randint(0,32)

    df = pd.read_csv(f'raw_data/clean_csvs/csv_{random_value}.csv')
    first_sequence = df['pitch_dur0'][:8]


    #-----convert the first sequence to list of notes----
    lis_first_sequence = list(first_sequence)
    first_input_sequence = []
    for note in lis_first_sequence:
        note = note.split('-')
        note[0] = int(note[0])
        first_input_sequence.append(note)


    return {'first_sequence': first_input_sequence} #before normalizing format

@app.get('/predict')
def predict(sequence):
    #-----grabbing the model-----
    model = keras.models.load_model("model.h5")

    #-----transform the sequence to the format model can take in-----
    #sequence example : [actual pitch, actual duration]
    list_sequence = ast.literal_eval(sequence)
    input_sequence = []

    for note in list_sequence:
        print(note)
        print(len_in_64th_notes)
        dur_len_in_64th_notes = len_in_64th_notes[note[1]]
        dur_mapped = duration_mapping[dur_len_in_64th_notes]
        pitch_mapped = pitch_mapping[note[0]]
        mapped_note = [pitch_mapped, dur_mapped]
        note_normalized = [mapped_note[0]/float(L_pitch_symb), mapped_note[1]/float(L_duration_symb)]
        input_sequence.append(note_normalized)

    #-----take in the sequence and predict-----
    input_sequence = np.array(input_sequence).reshape(1,8,2)
    prediction = model.predict(input_sequence)

    #-----transform the prediction into the notes the front-end can take in-----
    # return predictions from sample
    pitch_pred, duration_pred = prediction

    # apply randomness level to pitch
    temperature = 0.05  # randomness
    pitch_pred[0] /= temperature

    # get 3 random (weighted) indexes from top 12 pitch logits
    pitch_index_top_12 = np.argpartition(pitch_pred[0], -12)[-12:]
    pitch_logits_top_12 = np.array([pitch_pred[0][i] for i in pitch_index_top_12]).reshape(-1, 12)
    pitch_3_logit_ind = np.array(tf.random.categorical(pitch_logits_top_12, 3)).reshape(-1)
    pitch_index_top_3 = [pitch_index_top_12[i] for i in pitch_3_logit_ind]

    dur_index_top_2 = np.argpartition(duration_pred[0], -2)[-2:]

    # return three notes as [pitch, duration] pairs
    three_notes = [[pitch, np.random.choice(dur_index_top_2)] for pitch in pitch_index_top_3]
    three_notes_mapped = [[pitch_reverse_mapping[pitch], duration_reverse_mapping[duration]] for pitch, duration in three_notes]


    return {'predictions': three_notes_mapped}
