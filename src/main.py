import argparse
from io import open
import os
import time
import numpy as np

from future.builtins import range
from future.utils import iteritems

# from datautils import load_data, Lang
# from models.LogisticRegression import get_dataloader, Model

# from datautils import load_data, Lang
# from models.Perceptron import get_dataloader, Model

from sequenceDatautils import load_data, Lang
from models.RNN import get_dataloader, Model


def main():
    """
    Executes the baseline model. This loads the training data, training labels, and dev data, then trains a logistic
    regression model, then dumps predictions to the specified file.

    Modify the middle of this code, between the two commented blocks, to create your own model.
    """

    parser = argparse.ArgumentParser(description='Duolingo shared task baseline model')
    parser.add_argument('--train', help='Training file name', required=True)
    parser.add_argument('--dev', help='Dev file name, to make predictions on', required=True)
    parser.add_argument('--test', help='Test file name, to make predictions on', required=True)
    parser.add_argument('--devpred', help='Output file name for predictions, defaults to test_name.pred')
    parser.add_argument('--testpred', help='Output file name for predictions, defaults to test_name.pred')
    args = parser.parse_args()

    assert os.path.isfile(args.train)
    assert os.path.isfile(args.dev)
    assert os.path.isfile(args.test)

    # Assert that the train course matches the test course
    assert os.path.basename(args.train)[:5] == os.path.basename(args.test)[:5] == os.path.basename(args.dev)[:5]

    # ============================== Hyper Parameter ==============================
    dbg = True
    epochs = 1 if dbg else 10
    lang = Lang()

    # ============================== Data Loading ==============================

    print('Begin Data Loading')
    start_time = time.time()
    training_data, training_labels = load_data(args.train, lang, dbg)
    dev_data = load_data(args.dev, lang)
    test_data = load_data(args.test, lang)
    end_time = time.time()
    print('Data Loaded\t Time Taken %0.2fm' % ((end_time - start_time)/60))

    # ============================== Training ==============================
    model = Model(lang)

    print('Begin Training')
    train_loader = get_dataloader(training_data, training_labels, lang)
    model.train(train_loader, epochs)

    # ============================== Inference ==============================
    print('Begin Inference-Dev')
    dev_loader = get_dataloader(dev_data, np.zeros(len(dev_data)), lang)
    predictions = model.predict_for_set(dev_loader)
    with open(args.devpred, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')
    
    print('Begin Inference-Test')
    test_loader = get_dataloader(test_data, np.zeros(len(test_data)), lang)
    predictions = model.predict_for_set(test_loader)
    with open(args.testpred, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')

if __name__ == '__main__':
    main()
