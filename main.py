import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from datetime import date, datetime

num_classes = 11
input_dim = 30
train_epochs = 50


def read_datasets(path):
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        data = []
        for row in rows:
            data.append(row)
        print('load success from', path)
        return data[1:]


def split_data(train_data):
    x_train = train_data
    y_train = []
    print(len(train_data))
    for row in x_train:
        result = row.pop()
        y_train.append(result)
    return x_train, y_train


def process_data(x_train, x_test):
    new_x_train = []
    for data in x_train:
        new_data = []
        data = data[4:]
        for value in data:
            if value == '':
                new_data.append(0.0)
            else:
                new_data.append(float(value))
        new_x_train.append(new_data)

    new_x_test = []
    for data in x_test:
        new_data = []
        data = data[4:]
        for value in data:
            if value == '':
                new_data.append(0.0)
            else:
                new_data.append(float(value))
        new_x_test.append(new_data)

    return new_x_train, new_x_test


def train_model_and_predict(x_train, x_test, y_train):
    model = Sequential()
    model.add(Dense(units=256, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_history = model.fit(x=x_train, y=y_train, validation_split=0.0, epochs=train_epochs, batch_size=32, verbose=1)

    y_test = []
    predict_results = model.predict(x_test)
    for result in predict_results:
        y_test.append(np.argmax(result))
    return y_test


def write_submission_file(result):
    submission_rows = []
    with open('./data/submission.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            submission_rows.append(row)

    for i in range(len(result)):
        submission_rows[i+1][1] = result[i]

    now = datetime.now()
    path = './results/submission_' + now.strftime("%Y-%m-%d_%H:%M:%S") + '.csv'
    file = open(path, 'w')
    writer = csv.writer(file)
    for row in submission_rows:
        writer.writerow(row)
    file.close()


def main():
    print('Loading datasets...')
    train_data = read_datasets('data/train.csv')
    x_train, y_train = split_data(train_data)
    test_data = read_datasets('data/test.csv')
    x_test, y_test = test_data, []

    print(len(x_train), len(x_test), len(y_train), len(y_test))
    x_train, x_test = process_data(x_train, x_test)

    print('Reshaping data...')
    x_train = np.array(x_train, dtype=float)
    print('x_train', x_train.shape)
    x_test = np.array(x_test, dtype=float)
    print('x_test', x_test.shape)
    y_train = np.array(y_train, dtype=int)
    print('y_train', y_train.shape)

    y_train = np_utils.to_categorical(y_train, num_classes)

    print('Training data...')
    y_test = train_model_and_predict(x_train, x_test, y_train)
    print('Predict_result: ', y_test)

    write_submission_file(y_test)


if __name__ == '__main__':
    main()
