import pandas as pd
import music21 as m21
import glob


def import_midis(path):
    """creates a list of music21 midi files

    Args:
        path (str): path to folder that has midi files
    """
    filenames = glob.glob(path+'*')
    midi_list = [m21.converter.parse(file) for file in filenames]
    return midi_list


def inspect_melody_df(df):
    """shows sheet music and plays midi from a dataframe created with midi_to_melody_df function

    Args:
        df (pandas.DataFrame): one dataframe with pitch and duration columns"""

    s = m21.stream.Stream()
    for ind, note in df.iterrows():
        n = m21.note.Note(note.pitch, quarterLength=note.duration) if note.pitch != 0 else m21.note.Rest(quarterLength=note.duration)
        s.append(n)

    s.show()
    s.show('midi')

key_converter_dict = {'g- major': 6, 'g major': 5, 'a- major': 4, 'a major': 3,
                      'b- major': 2, 'b major': 1, 'c major': 0, 'd- major': -1,
                      'd major': -2, 'e- major': -3, 'e major': -4, 'f major': -5, 'f# major': -6,
                      'e- minor': 6, 'e minor': 5, 'f minor': 4, 'f# minor': 3,
                      'g minor': 2, 'g# minor': 1, 'a minor': 0, 'b- minor': -1,
                      'b minor': -2, 'c minor': -3, 'c# minor': -4, 'd minor': -5, 'd# minor': -6,
                      }


def midi_to_melody_df(midi_list):
    """converts list of m21 parsed midis into list of melody dataframes, where each row is one note,
    with columns for pitch, duration, beat, and measure.

    note:
        - if there is no key signature included in the midi file, the sheet music will be shown and a prompt to type the key signature will appear
        - tuplets and unusual rhythms are removed
        - pitch: in MIDI numbers
        - duration: in quarterLengths
        - rests: appear as pitch 0

    Args:
        midi_list(list): list of midis created with import_midis function
    """
    columns = ['pitch', 'duration', 'beat', 'measure']
    note_dfs = []

    for midi in midi_list:
        # get notes from midis
        notes = []
        key_sig = ""
        pick = midi.parts[0].recurse()
        for element in pick:
            if '/' in str(element.duration.quarterLength) or '/' in str(element.beat):
                notes.append([0, 0, 0, 0])
                continue
            if isinstance(element, m21.note.Note):
                notes.append([int(element.pitch.midi), float(element.duration.quarterLength), float(element.beat), int(element.measureNumber)])
            if isinstance(element, m21.note.Rest):
                notes.append([int(0), float(element.duration.quarterLength), float(element.beat), int(element.measureNumber)])
            if isinstance(element, m21.chord.Chord):
                notes.append([max([n.midi for n in element.pitches]), float(element.duration.quarterLength), float(element.beat), int(element.measureNumber)])
            if isinstance(element, m21.key.Key):
                key_sig = str(element).lower()

        # transpose to c major / a minor
        if key_sig == "":
            midi.show()
            key_sig = input("enter the key signature of the sheet music (eg a minor, b- major, c# minor, etc): ").lower()
            while key_sig[1] not in ['#', '-', ' '] and key_sig[0] not in ['a', 'b', 'c', 'd', 'e', 'f', 'g'] and key_sig[-5:] not in ['major', 'minor']:
                key_sig = input("please enter a valid key signature: ")

        for note in notes:
            if note[0] == 0:
                continue
            note[0] += key_converter_dict[key_sig]

        # create df and add to list
        note_dfs.append(pd.DataFrame(notes, columns=columns))

    return note_dfs


def split_melody_dfs(df_list):
    """split melody dfs into separate dataframes based on where measure is 0
    (ie where there were tuplets / strange rhythms).
    It will only add the dataframe if it has 2 or more rows.

    Args:
        df_list (list): a list of dataframes made with the midi_to_melody_df function
    """
    split_dfs = []

    for df in df_list:
        prev_0 = -1
        for i, val in df.iterrows():
            if val.measure == 0:
                new_df = df[prev_0+1: i].reset_index(drop=True)
                if len(new_df) > 1:
                    split_dfs.append(new_df)
                prev_0 = i

    return split_dfs


if __name__ == "__main__":
    path = '../raw_data/mozart_midis/'
    midi_list = import_midis(path)
    print('midis put in midi_list')
    note_dfs = midi_to_melody_df(midi_list)
    print('note_dfs created')
    split_dfs = split_melody_dfs(note_dfs)
    print('note_dfs split')
