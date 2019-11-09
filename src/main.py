import argparse
from io import open
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from future.builtins import range
from future.utils import iteritems

from datautils import load_data
from models.MLP import get_dataloader, Model

epochs = 1

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

    training_data, training_labels = load_data(args.train)
    test_data = load_data(args.test)

    ####################################################################################
    # Here is the delineation between loading the data and running the baseline model. #
    # Replace the code between this and the next comment block with your own.          #
    ####################################################################################

    torch.manual_seed(23)
    np.random.seed(23)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.L1Loss()

    # ============================== Train ==============================
    model.train()
    model.to(device)

    running_loss = 0.0
    dataloader = get_dataloader(training_data, training_labels)
    
    start_time = time.time()
    for (feats, labels) in dataloader:
        optimizer.zero_grad() 
        feats = feats.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()

        outputs = model(feats)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    running_loss /= len(dataloader)
    print('Training Time: %0.2f min' % ((end_time - start_time)/60))
    print('\tTraining Loss: ', running_loss)
    
    # ============================== Inference ==============================
    model.eval()
    model.to(device)

    predictions = dict()
    dataloader = get_dataloader(test_data, np.zeros(len(test_data)))
    
    start_time = time.time()
    with torch.no_grad():
        for (feats, ids) in dataloader:
            optimizer.zero_grad() 
            feats = feats.to(device, non_blocking=True).float()
            outputs = model(feats)
            for i in range(len(feats)):
                predictions[ids[i]] = outputs[i].item()
    
    end_time = time.time()
    
    print('Inference Time: %0.2f min' % ((end_time - start_time)/60))

    ####################################################################################
    # This ends the baseline model code; now we just write predictions.                #
    ####################################################################################

    with open(args.pred, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')

if __name__ == '__main__':
    main()
