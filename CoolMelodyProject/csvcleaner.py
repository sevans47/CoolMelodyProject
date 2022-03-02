import pandas as pd
import numpy as np

# from csvcombiner import get_movement_df_list, get_movement_filenames

# movement_dfs = get_movement_df_list()
# print(movement_dfs[0])

### set up dictionaries

# movement key: # of half steps to add to get to c major / a minor
maj_key_converter_dict = {'gb major': 6, 'g major': 5, 'ab major': 4, 'a major': 3,
                          'bb major': 2, 'b major': 1, 'c major': 0, 'db major': -1,
                          'd major': -2, 'eb major': -3, 'e major': -4, 'f major': -5, 'f# major': -6}
min_key_converter_dict = {'eb minor': 6, 'e minor': 5, 'f minor': 4, 'f# minor': 3,
                          'g minor': 2, 'g# minor': 1, 'a minor': 0, 'bb minor': -1,
                          'b minor': -2, 'c minor': -3, 'c# minor': -4, 'd minor': -5, 'd# minor': -6}

# bpm note value: amount to multiply bpm by to convert bpm to quarter notes
bpm_converter_dict = {'2': 2, '2d': 3, '4': 1, '4d': 1.5, '8': 0.5}

# note duration: amount to multiply spb (seconds per beat) by to get duration in seconds
beat_dict = {'1': 4, '2': 2, '4': 1, '8': 0.5,'16': 0.25,'32': 0.125,'64': 0.0625}

def normalize_pitches(note, piece_key_type, piece_key):
    """change the pitch of notes so that they're in the key of c major / a minor"""
    if note > 0:
        if piece_key_type == 'major':
            note += maj_key_converter_dict[piece_key]
        else:
            note += min_key_converter_dict[piece_key]
    return note

def duration_to_seconds(note, piece_spb):
    """change the duration of notes so that they're in seconds"""
    if type(note) != 'str':
        note = str(note)
    if note[-2:] == 'dd':
        s = piece_spb * beat_dict[note[:-2]] * 1.75
    elif x[-1] == 'd':
        s = piece_spb * beat_dict[note[:-1]] * 1.5
    else:
        s = piece_spb * beat_dict[note]
    return s
