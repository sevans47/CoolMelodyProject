import pandas as pd
import numpy as np
import pretty_midi
import collections

from CoolMelodyProject.csvcombiner import get_movement_filenames


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
    elif note[-1] == 'd':
        s = piece_spb * beat_dict[note[:-1]] * 1.5
    else:
        s = piece_spb * beat_dict[note]
    return s


def process_df(filename: str) -> pd.DataFrame:

    """from the csv's filename, creates a df with pitch, pitch_norm, and dur(s) columns added"""

    # import pieces df
    pieces_df = pd.read_csv('../raw_data/mozart_sonatas/mps - pieces.csv')

    # extract the K number
    k_num = filename.split(' - ')[1].split('.csv')[0].strip(' ')

    # create df
    path = '../raw_data/mozart_sonatas/'
    df = pd.read_csv(path + filename)  # make df

    # normalize pitch
    df['pitch'] = df.note_name.apply(lambda x: pretty_midi.note_name_to_number(x) if x != 'r' else 0)
    piece_key = pieces_df[pieces_df['name'] == k_num]['key'].values[0].strip(' ')
    piece_key_type = piece_key.split(' ')[1].strip(' ')
    df['pitch_norm'] = df.pitch.apply(normalize_pitches, args=(piece_key_type, piece_key))

    # get duration in seconds
    piece_bpm = pieces_df[pieces_df['name'] == k_num]['bpm'].values[0].split('=')
    piece_bpm_norm = int(bpm_converter_dict[piece_bpm[0]] * int(piece_bpm[1]))
    piece_spb = 60 / piece_bpm_norm
    df['dur(s)'] = df.duration.apply(duration_to_seconds, args=(piece_spb,))

    return df


def notes_to_midi(filename: str) -> pretty_midi.PrettyMIDI:

    """from the csv's filename, create a pretty midi object with the midi data for the piece"""

    notes = process_df(filename)

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            'Acoustic Guitar (nylon)'))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start)
        end = float(prev_start + note['dur(s)'])
        if note['note_name'] == 'r':
            start += float(note['dur(s)'])
            prev_start = start
        else:
            note = pretty_midi.Note(
                velocity=100,
                pitch=notes['pitch_norm'],
                start=start,
                end=end
            )
            instrument.notes.append(note)
            prev_start = end

    pm.instruments.append(instrument)
    return pm


def midi_to_notes(filename: str) -> pd.DataFrame:

    """from the csv's filename, create a clean dataframe for training"""

    midi = notes_to_midi(filename)

    instrument = midi.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start-prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


if __name__ == "__main__":
    filenames = get_movement_filenames()
    print(filenames[0])
    # example_pm = notes_to_midi(filenames[0])
    # instrument = example_pm.instruments[0]
    df = midi_to_notes(filenames[0])
    print(df.head())
