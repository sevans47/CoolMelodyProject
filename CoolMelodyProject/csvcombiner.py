import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


###Set up necessary variables

mypath = '../raw_data/mozart_sonatas' #folder of pieces
metadata_filenames = ['documentation.txt','mps - pieces.csv'] #files we want to ignore for getting just indiv pieces
pieces_metadata_filepath = join(mypath,'mps - pieces.csv') #filepath for the pieces.csv file



def get_movement_filenames(mypath='../raw_data/mozart_sonatas', metadata_filenames = ['documentation.txt','mps - pieces.csv']):
    '''make a list of file names in the data folder'''

     #files we want to ignore for getting just indiv pieces

    movement_filenames = [filename for filename in listdir(mypath) if isfile(join(mypath, filename))]
    #listdir gives list of files and directories at path end. if statement will check if its a file so we ignore dirs

    for filename in metadata_filenames:  #remove metadata filenames
        if filename in movement_filenames:    #make sure no error if already removed
            movement_filenames.remove(filename)

    return movement_filenames


def get_movement_filepaths():
    '''make a list of file paths in the data folder'''

    movement_filenames = get_movement_filenames()
    movement_filepaths = [join(mypath, filename) for filename in movement_filenames]
    return movement_filepaths

def get_movement_names():
    '''make a list of each movements name'''

    movement_filenames = get_movement_filenames()
    movement_names = [filename.split('mps - ')[1].split('.csv')[0] for filename in movement_filenames]
    return movement_names



def get_movement_df_dict():
    '''returns a dictionary with key = movement name, and value = movement dataframe'''

    movement_names = get_movement_names()
    movement_filepaths = get_movement_filepaths()

    movement_df_dict = {name:pd.read_csv(path) for path, name in zip(movement_filepaths,movement_names)}
    for key in movement_df_dict:
        movement_df_dict[key]['movement_name'] = key

    return movement_df_dict


def get_movement_df_list():
    '''returns a list of movement dataframes'''

    movement_names = get_movement_names()
    movement_filepaths = get_movement_filepaths()

    movement_df_list = [pd.read_csv(path) for path in movement_filepaths]
    for i in range(len(movement_df_list)):
        movement_df_list[i]['movement_name'] = movement_names[i]

    return movement_df_list



def get_stacked_movement_df():


    movement_df_list = get_movement_df_list()

    stacked = pd.concat(movement_df_list)
    stacked.reset_index(inplace = True)
    stacked.rename(columns={'index':'note_num'}, inplace=True)

    return stacked


def export_stacked():

    stacked = get_stacked_movement_df()

    path = '../raw_data/'
    name = 'stacked_movement_df.csv'
    stacked.to_csv(join(path,name)) #will overwrite curent file if same name so can be rerun

if __name__ == '__main__':
    print(get_stacked_movement_df())
