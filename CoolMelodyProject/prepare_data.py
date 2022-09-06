import pandas as pd
import numpy as np
import json
from midi_cleaner import import_midis, midi_to_melody_df, split_melody_dfs

MIDI_PATH = '../raw_data/mozart_midis/'
JSON_PATH = 'data/data.json'
SEQUENCE_LENGTH = 8

def midi_to_dfs(midi_path):

    # import midi files
    midi_list = import_midis(midi_path)

    # extract melody and save as dataframes
    midi_dfs = midi_to_melody_df(midi_list)

    # clean the melodies
    note_dfs = split_melody_dfs(midi_dfs)

    return note_dfs


def create_mapping_dicts(note_dfs):

    # create pitch / duration corpora
    pitch_corpus = []
    duration_corpus = []

    for df in note_dfs:
        pitch_corpus += list(df['pitch'])
        duration_corpus += list(df['duration'])

    # Storing all the unique pitches / durations present in the corpus to buid a mapping dict.
    pitch_symb = sorted(list(set(pitch_corpus)))
    duration_symb = sorted(list(set(duration_corpus)))

    L_pitch_corpus = len(pitch_corpus)  # length of corpus
    L_duration_corpus = len(duration_corpus)
    L_pitch_symb = len(pitch_symb)  # length of total unique characters
    L_duration_symb = len(duration_symb)

    # Building dictionary to access the pitches / durations from indices and vice versa
    pitch_mapping = dict((int(c), i) for i, c in enumerate(pitch_symb))
    pitch_reverse_mapping = dict((i, int(c)) for i, c in enumerate(pitch_symb))
    duration_mapping = dict((float(c), i) for i, c in enumerate(duration_symb))
    duration_reverse_mapping = dict((i, float(c)) for i, c in enumerate(duration_symb))

    # adding padding value to the dictionaries
    pitch_mapping[-1] = -1
    pitch_reverse_mapping[-1] = -1
    duration_mapping[-1.0] = -1
    duration_reverse_mapping[-1] = -1.0

    return pitch_mapping, pitch_reverse_mapping, duration_mapping, duration_reverse_mapping


def mask_start_df(df, pad_length):

    # add pad_length number of -1's to the start of each df
    mask_df = pd.DataFrame(np.full((pad_length, 4), -1), dtype = 'int64')
    mask_df.columns = ['pitch','duration','beat','measure']
    return pd.concat([mask_df, df])


def create_sequences(df_list, length, pitch_mapping, duration_mapping, horizon = 1, selected_features = ['pitch', 'duration']):
    """
    Split the df's into sequences of equal length (X) and output target (y_p, y_d)

    Args:
    - df_list           (list)  : note_dfs (list of dfs made with midi_to_dfs function)
    - length            (int)   : length of each sequence
    - pitch_mapping     (dict)  : dictionary mapping pitch number to integers
    - duration_mapping  (dict)  : dictionary mapping duration values to integers
    - horizon           (int)   : how many notes past the desired sequence length we will accept for dataframes.
                                  Default is 1, which is length of target
    - selected_features (list)  : which features from the dataframe that sequences will be created sequences for
    """
    features_list = []
    target_pitch = []
    target_duration = []
    for note_df in df_list:
        df = note_df.copy()
        df = mask_start_df(df, pad_length = (length // 2)) # add padding the to start of each data frame
        L_df = len(df)
        if L_df >= (length + horizon):
            df = df.reset_index()
            df['pitch'] = df['pitch'].astype('int') # to match dictionary keys
            df['duration'] = df['duration'].astype('float') # to match dictionary keys
            df['pitch'] = df['pitch'].map(pitch_mapping)
            df['duration'] = df['duration'].map(duration_mapping)
            latest_start_index = (L_df - length - horizon)
            for i in range(latest_start_index):

                features = df.loc[i:(i + length - 1), selected_features] # minus one to exclude target
                features_list.append(features)

                pitch = df.loc[(i + length), 'pitch']
                target_pitch.append(pitch)

                duration = df.loc[(i + length), 'duration']
                target_duration.append(duration)

    # L_datapoints = len(target_pitch)
    # print("Total number of sequences in the dataset:", L_datapoints)

    return np.array(features_list), np.array(target_pitch), np.array(target_duration)


def remove_repeat_sequences(X, y_p, y_d, threshold = 25):
    """
    If the same sequence appears more times than the threshold, remove any excess sequences past the threshold.
    """
    _, unique_indexes, counts = np.unique(X, axis = 0, return_index = True, return_counts = True)
    indexes = [idx for idx, count in zip(unique_indexes, counts) if count < threshold]
    return X[indexes], y_p[indexes].reshape(-1, 1), y_d[indexes].reshape(-1, 1)


def create_sample_weights(y_pitch, y_duration):

    # get unique values for pitch / duration and their number of occurences
    pitch_counts = np.unique(y_pitch, return_counts=True)
    duration_counts = np.unique(y_duration, return_counts=True)

    # make dict - pitch / duration value: counts
    pitch_class_weights_simple = {key:val for key, val in zip(pitch_counts[0], pitch_counts[1])}
    duration_class_weights_simple = {key:val for key, val in zip(duration_counts[0], duration_counts[1])}

    # find average count for pitch and duration
    pitch_mean = np.mean(list(pitch_class_weights_simple.values()))
    duration_mean = np.mean(list(duration_class_weights_simple.values()))

    # create class weights - (1 / count) * avg count
    pcw = {key:(1 / val) * pitch_mean for key, val in zip(pitch_counts[0], pitch_counts[1])}
    dcw = {key:(1 / val) * duration_mean for key, val in zip(duration_counts[0], duration_counts[1])}

    # create class weight of 0 for values that are in X but not in y
    pcw_keys = list(pcw)
    dcw_keys = list(dcw)

    for i in range(int(max(pcw_keys))):
        if i not in pcw_keys:
            pcw[i] = 0

    for i in range(int(max(dcw_keys))):
        if i not in dcw_keys:
            dcw[i] = 0

    # create arrays that replace y values with class weight values for y
    y_pitch_cw = np.array([pcw[pitch] for pitch in y_pitch[:, 0]])
    y_dur_cw = np.array([dcw[dur] for dur in y_duration[:, 0]])

    # take average value of class weights for y_p and y_d for each sample
    sample_weights = ((y_pitch_cw + y_dur_cw) / 2).reshape(-1, 1)

    return sample_weights


def main(midi_path, sequence_length, json_path):

    # load midi files and store as list of dataframes
    print("loading midis ...")
    note_dfs = midi_to_dfs(midi_path)

    # create duration and pitch mapping dictionaries from dfs
    pitch_mapping, pitch_reverse_mapping, duration_mapping, duration_reverse_mapping = create_mapping_dicts(note_dfs)

    # create sequences of notes and target pitch and duration for model
    print("creating sequences ...")
    X, y_p, y_d = create_sequences(note_dfs, sequence_length, pitch_mapping, duration_mapping)

    # remove sequences that appear above threshold
    X, y_pitch, y_duration = remove_repeat_sequences(X, y_p, y_d)

    # create sample weights
    print("creating sample weights ...")
    sample_weights = create_sample_weights(y_pitch, y_duration)

    # save data as json file
    data_dict = {
        'X': X.tolist(),
        'y_pitch': y_pitch.tolist(),
        'y_duration': y_duration.tolist(),
        'sample_weights': sample_weights.tolist(),
        'pitch_mapping': pitch_mapping,
        'pitch_reverse_mapping': pitch_reverse_mapping,
        'duration_mapping': duration_mapping,
        'duration_reverse_mapping': duration_reverse_mapping
    }

    with open(json_path, 'w') as f:
        json.dump(data_dict, f, indent=4)

    print("data saved as .json")


if __name__ == "__main__":
    main(MIDI_PATH, SEQUENCE_LENGTH, JSON_PATH)
