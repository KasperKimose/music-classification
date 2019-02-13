import argparse
import os

import librosa
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split

import numpy as np

import mp3_to_npy_convertor

song_samples = 660000
#genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
         # 'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
genres = {'metal': 0, 'disco': 1}

def main():
    # print(args)

    gtzan_dir = '../genres/'
    # gtzan_dir = args.directory

    load_from_file = False

    x, y = None, None
    if load_from_file:
        x, y = load_data_from_file('x_gtzan_npy.npy', 'y_gtzan_npy.npy')
    else:
        # Read the data
        x, y = read_data(gtzan_dir, genres, song_samples, to_melspectrogram, debug=True)

        np.save('x_gtzan_npy.npy', x)
        np.save('y_gtzan_npy.npy', y)

    model = build_model(x)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # One hot encoding of the labels
    y = to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    model.fit(x_train, y_train,
              batch_size=32,
              epochs=5,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))


def load_data_from_file(x_data_file, y_data_file):
    return np.load(x_data_file), np.load(y_data_file)

def build_model(songs):
    input_shape = songs[0].shape
    num_genres = 2

    model = Sequential()

    # First conv block
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Second conv block
    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Third conv block
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Fourth conv block
    model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Fifth conv block
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
    model.add(Dropout(0.25))

    # MLP
    model.add(Flatten())
    model.add(Dense(num_genres, activation='softmax'))

    model.summary()

    return model


def read_data(src_dir, genres, song_samples, spec_format, debug=False):
    arr_specs = []
    arr_genres = []

    for genre,_ in genres.items():
        folder = src_dir + genre

        for root, sub_dirs, files in os.walk(folder):
            for file in files:
                # Read the song
                filename = folder + '/' + file
                signal, sr = librosa.load(filename)
                signal = signal[:song_samples]

                if debug:
                    print('Read file: ' + filename)

                # Convert song to sub songs
                sub_signals, sub_genres = split_song(signal, genres[genre])

                # Convert to the chosen format
                transformed_sub_signals = spec_format(sub_signals)

                arr_genres.extend(sub_genres)
                arr_specs.extend(transformed_sub_signals)

    return np.array(arr_specs), np.array(arr_genres)


def split_song(song, genre, window=0.1, overlap=0.5):
    # Empty list to hold data
    temp_song = []
    temp_genre = []

    # Get the input songs array size
    x_shape = song.shape[0]
    chunk = int(x_shape * window)
    offset = int(chunk * (1. - overlap))

    # Split song and create sub samples
    splitted_song = [song[i:i + chunk] for i in range(0, x_shape - chunk + offset, offset)]
    for sub_song in splitted_song:
        temp_song.append(sub_song)
        temp_genre.append(genre)

    return np.array(temp_song), np.array(temp_genre)


"""
@description: Method to convert a list of songs to a np array of melspectrograms
"""
def to_melspectrogram(songs, n_fft = 1024, hop_length = 512):
    mel_spec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft, hop_length=hop_length)[:, :, np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    transformed_song = map(mel_spec, songs)
    return np.array(list(transformed_song))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Music Genre Recognition on GTZAN data set')
    #
    # parser.add_argument('-d', '--directory', help('Path to the directory containing music file', type=str))
    #
    # args = parser.parse_args()
    #
    # if not args.directory:
    #     print('Please input directory for music data')
    #     exit()

    #main()
    #mp3_to_npy_convertor.convert_files("C:\\Users\\kkr\\Desktop\\Thesis\\audio_A", "../npys/", 22050, 640512)
    mp3_to_npy_convertor.convert_files("./", "../npys/", 22050, 640512)
