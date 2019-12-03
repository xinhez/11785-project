import argparse
from io import open
import os
import time
import numpy as np

from future.builtins import range
from future.utils import iteritems

# from datautils.datautils import load_data, Lang
# from models.LogisticRegression import get_dataloader, Model

# from datautils.datautils import load_data, Lang
# from models.Perceptron import get_dataloader, Model

from datautils.seqDatautils import load_data, Lang
from models.Seq2seq import get_dataloader, Model


def main():
    """
    Executes the baseline model. This loads the training data, training labels, and dev data, then trains a logistic
    regression model, then dumps predictions to the specified file.

    Modify the middle of this code, between the two commented blocks, to create your own model.
    """

    parser = argparse.ArgumentParser(description='Duolingo shared task baseline model')
    parser.add_argument('--language', help='choose from [es_en, en_es, fr_en]', required=True)
    parser.add_argument('--dataset_path', default='../data/%s/', required=False)
    parser.add_argument('--outputs_path', default='./outputs/', required=False)
    args = parser.parse_args()

    dataset_path = args.dataset_path % args.language
    assert os.path.isdir(dataset_path)

    train_path = dataset_path + '%s.slam.20190204.train' % args.language
    dev_path   = dataset_path + '%s.slam.20190204.dev'   % args.language
    test_path  = dataset_path + '%s.slam.20190204.test'  % args.language
    assert os.path.isfile(train_path)
    assert os.path.isfile(dev_path)
    assert os.path.isfile(test_path)

    if not os.path.isdir(args.outputs_path): os.mkdir(args.outputs_path)

    # ============================== Hyper Parameter ==============================
    dbg = True
    from_path = None
    epochs = 10 if dbg else 10
    lang = Lang()

    # ============================== Data Loading ==============================

    print('Begin Data Loading')
    start_time = time.time()
    training_data, training_labels = load_data(train_path, lang, dbg=dbg)
    dev_data  = load_data(dev_path,  lang)
    test_data = load_data(test_path, lang)
    end_time = time.time()
    print('Data Loaded\t Time Taken %0.2fm' % ((end_time - start_time)/60))

    # ============================== Training ==============================
    model = Model(lang)

    print('Begin Training')
    train_loader = get_dataloader(training_data, lang, training_labels)
    model.train(train_loader, epochs)

    # ============================== Inference ==============================
    print('Begin Inference-Dev', end=' ')
    start_time = time.time()
    dev_loader = get_dataloader(dev_data, lang)
    predictions = model.predict_for_set(dev_loader, from_path)
    with open(args.outputs_path + '%s_dev_predictions.pred' % args.language, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')
    end_time = time.time()
    print('| %0.2fm' % ((end_time-start_time)/60))
    
    print('Begin Inference-Test', end=' ')
    start_time = time.time()
    test_loader = get_dataloader(test_data, lang)
    predictions = model.predict_for_set(test_loader, from_path)
    with open(args.outputs_path + '%s_test_predictions.pred' % args.language, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')
    end_time = time.time()
    print('| %0.2fm' % ((end_time-start_time)/60))

if __name__ == '__main__':
    main()
