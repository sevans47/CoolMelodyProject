import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping

JSON_PATH = "data/data.json"
PLOT_PATH = "data/learning_curves.png"
LEARNING_RATE = 0.001
PATIENCE = 10

def load_data(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)

    # extract X, y, and sample weights
    X = np.array(data["X"])
    y_pitch = np.array(data["y_pitch"])
    y_duration = np.array(data["y_duration"])
    y = {"pitch_output": y_pitch, "duration_output": y_duration}
    sample_weights = np.array(data["sample_weights"])

    # extract mapping dictionaries
    pitch_mapping = data["pitch_mapping"]
    pitch_reverse_mapping = data["pitch_reverse_mapping"]
    duration_mapping = data["duration_mapping"]
    duration_reverse_mapping = data["duration_reverse_mapping"]

    mapping_dicts = [pitch_mapping, pitch_reverse_mapping, duration_mapping, duration_reverse_mapping]

    return X, y, sample_weights, mapping_dicts


def build_model(input_shape, learning_rate, num_pitches, num_durations):

    # set optimizer and metric
    opt = Adam(learning_rate=learning_rate)
    metric = SparseTopKCategoricalAccuracy(k=3, name='sparse_top_k_categorical_accuracy')

    # build model
    input_layer = Input(shape=input_shape, name='input_layer')
    first_LSTM = LSTM(32, name='first_LSTM', activation='tanh', return_sequences=True)(input_layer)
#     first_dropout = Dropout(0.1, name='first_dropout')(first_LSTM)

    pitch_LSTM = LSTM(16, name='pitch_LSTM', activation='tanh')(first_LSTM)
    pitch_dropout = Dropout(0.2, name='pitch_dropout')(pitch_LSTM)
    pitch_dense = Dense(16, name='pitch_dense', activation='tanh')(pitch_dropout)
    pitch_dropout_2 = Dropout(0.4, name='pitch_dropout_2')(pitch_dense)
    pitch_output = Dense(num_pitches, name='pitch_output', activation='softmax')(pitch_dropout_2)


    duration_LSTM = LSTM(16, name='duration_LSTM', activation='tanh')(first_LSTM)
    duration_dropout = Dropout(0.2, name='duration_dropout')(duration_LSTM)
    duration_dense = Dense(16, name='duration_dense', activation='tanh')(duration_dropout)
    duration_dropout_2 = Dropout(0.4, name='duration_dropout_2')(duration_dense)
    duration_output = Dense(num_durations, name='duration_output', activation='softmax')(duration_dropout_2)

    model = Model(inputs=input_layer, outputs=[pitch_output, duration_output])

    losses = {
        "pitch_output": "sparse_categorical_crossentropy",
        "duration_output": "sparse_categorical_crossentropy",
    }

    loss_weights = {
        "pitch_output": 1.0,
        "duration_output": 1.0
    }

    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=metric)

    return model


def plot_curves(history, plot_path):

    fig, axes = plt.subplots(2, 3, figsize = (12, 5))

    axes[0, 0].plot(history['loss'], label = 'Train')
    axes[0, 0].plot(history['val_loss'], label = 'Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(history['pitch_output_loss'], label = 'Train')
    axes[0, 1].plot(history['val_pitch_output_loss'], label = 'Val')
    axes[0, 1].set_title('Pitch Loss')
    axes[0, 1].legend()

    axes[0, 2].plot(history['duration_output_loss'], label = 'Train')
    axes[0, 2].plot(history['val_duration_output_loss'], label = 'Val')
    axes[0, 2].set_title('Duration Loss')
    axes[0, 2].legend()

    # axes[1, 0].plot(history['sparse_top_k_categorical_accuracy'])
    # axes[1, 0].plot(history['val_sparse_top_k_categorical_accuracy'])
    # axes[1, 0].set_title('Top K Acc')
    # axes[1, 0].legend();

    axes[1, 1].plot(history['pitch_output_sparse_top_k_categorical_accuracy'], label = 'Train')
    axes[1, 1].plot(history['val_pitch_output_sparse_top_k_categorical_accuracy'], label = 'Val')
    axes[1, 1].set_title('Pitch Top K Acc')
    axes[1, 1].legend()


    axes[1, 2].plot(history['duration_output_sparse_top_k_categorical_accuracy'], label = 'Train')
    axes[1, 2].plot(history['val_duration_output_sparse_top_k_categorical_accuracy'], label = 'Val')
    axes[1, 2].set_title('Duration Top K Acc')
    axes[1, 2].legend()
    plt.tight_layout()

    plt.savefig(plot_path)


def main(json_path, learning_rate, patience, plot_path):

    # load data
    print("loading data ... ")
    X, y, sample_weights, mapping_dicts = load_data(json_path)

    # build model
    print("building model ... ")
    input_shape = X.shape[1:]
    num_pitchs = len(mapping_dicts[0])
    num_durations = len(mapping_dicts[2])
    model = build_model(input_shape, learning_rate, num_pitchs, num_durations)

    # train model
    es = EarlyStopping(patience=patience, restore_best_weights=True)

    print("training model ... ")
    history = model.fit(X,
                    y,
                    validation_split=0.2,
                    shuffle=True,
                    batch_size=16,
                    epochs=500,
                    verbose=0,
                    callbacks=[es],
                    sample_weight=sample_weights).history

    print("plotting curves ... ")
    plot_curves(history, plot_path)

    # show validation set scores for pitch and accuracy
    best_epoch = len(history['val_pitch_output_sparse_top_k_categorical_accuracy']) - 1 - patience
    best_pitch = history['val_pitch_output_sparse_top_k_categorical_accuracy'][best_epoch]
    best_duration = history['val_duration_output_sparse_top_k_categorical_accuracy'][best_epoch]

    print("="*50)
    print(f"Validation top k categorical accuracy for pitch: {best_pitch}")
    print(f"Validation top k categorical accuracy for duration: {best_duration}")

    print("complete")

if __name__ == "__main__":
    main(JSON_PATH, LEARNING_RATE, PATIENCE, PLOT_PATH)
