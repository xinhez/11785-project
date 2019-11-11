import argparse
from io import open
import os
import time
import torch
import torch.nn as nn
import numpy as np

from future.builtins import range
from future.utils import iteritems

from datautils import load_data
from models.MLP import get_dataloader, Model

epochs = 10

def main():
    """
    Executes the baseline model. This loads the training data, training labels, and dev data, then trains a logistic
    regression model, then dumps predictions to the specified file.

    Modify the middle of this code, between the two commented blocks, to create your own model.
    """

    parser = argparse.ArgumentParser(description='Duolingo shared task baseline model')
    parser.add_argument('--train', help='Training file name', required=True)
    parser.add_argument('--test', help='Test file name, to make predictions on', required=True)
    parser.add_argument('--pred', help='Output file name for predictions, defaults to test_name.pred')
    args = parser.parse_args()

    if not args.pred:
        args.pred = args.test + '.pred'

    assert os.path.isfile(args.train)
    assert os.path.isfile(args.test)

    # Assert that the train course matches the test course
    assert os.path.basename(args.train)[:5] == os.path.basename(args.test)[:5]

    start_time = time.time()
    training_data, training_labels = load_data(args.train)
    test_data = load_data(args.test)
    end_time = time.time()
    print('Data Loaded\t Time Taken %0.2fm' % ((end_time - start_time)/60))

    ####################################################################################
    # Here is the delineation between loading the data and running the baseline model. #
    # Replace the code between this and the next comment block with your own.          #
    ####################################################################################

    torch.manual_seed(0)
    np.random.seed(0)
    
    model = Model()
    train_loader = get_dataloader(training_data, training_labels)
    model.train(train_loader, epochs)
    
    test_loader = get_dataloader(test_data, np.zeros(len(test_data)))
    predictions = model.predict_test_set(test_loader)

    ####################################################################################
    # This ends the baseline model code; now we just write predictions.                #
    ####################################################################################

    with open(args.pred, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')

if __name__ == '__main__':
    main()
