# Duolingo SLAM Shared Task

This archive contains a baseline model and evaluation script for Duolingo's 2018 Shared Task on Second Language Acquisition Modeling (SLAM). 
The model is L2-regularized logistic regression, trained with SGD weighted by frequency.   

## Setup

This baseline model is written in Python. It depends on the `future` library for compatibility with both Python 2 and 3,
which on many machines may be obtained by executing `pip install future` in a console.

In order to run the baseline model and evaluate your predictions, perform the following:

* Download and extract the contents of the file to a local directory.
* To train the baseline model: 
  * Open a console and `cd` into the directory where `baseline.py` is stored
  * Execute: 
    
    ```bash
    python baseline.py --train path/to/train/data.train 
                       --test path/to/dev_or_test/data.dev
                       --pred path/to/dump/predictions.pred
    ``` 
    to create predictions for your chosen track. Note that we use `test` interchangeably for the dev and test sets because both are test sets.
* To evaluate the baseline model:
  * Execute     
  
    ```bash
    python eval.py --pred path/to/your/predictions.pred
                   --key path/to/dev_or_test/labels.dev.key
    ```
    to print a variety of metrics for the baseline predictions to the screen.

## Example Command
* es_en
    ```bash
    python3 officialBaseline.py --train ../data/es_en/es_en.slam.20190204.train --test ../data/es_en/es_en.slam.20190204.test --pred ./es_en_predictions.pred
    python3 eval.py --pred ./es_en_predictions.pred --key ../data/es_en/es_en.slam.20190204.test.key
    ```
* fr_en
    ```bash
    python3 officialBaseline.py --train ../data/fr_en/fr_en.slam.20190204.train --test ../data/fr_en/fr_en.slam.20190204.test --pred ./fr_en_predictions.pred
    python3 eval.py --pred ./fr_en_predictions.pred --key ../data/fr_en/fr_en.slam.20190204.test.key
    ```
* en_es
    ```bash
    python3 officialBaseline.py --train ../data/en_es/en_es.slam.20190204.train --test ../data/en_es/en_es.slam.20190204.test --pred ./en_es_predictions.pred
    python3 eval.py --pred ./en_es_predictions.pred --key ../data/en_es/en_es.slam.20190204.test.key
    ```

## Official Baseline Model
This baseline model loads the training and test data that you pass in via --train and --test arguments for a particular
track (course), storing the resulting data in InstanceData objects, one for each instance. The code then creates the
features we'll use for logistic regression, storing the resulting LogisticRegressionInstance objects, then uses those to
train a regularized logistic model with SGD, and then makes predictions for the test set and dumps them to a CSV file
specified with the --pred argument, in a format appropriate to be read in and graded by the [eval.py](eval.py) script.

We elect to use two different classes, InstanceData and LogisticRegressionInstance, to delineate the boundary between
the two purposes of this code; the first being to act as a user-friendly interface to the data, and the second being to
train and run a baseline model as an example. Competitors may feel free to use InstanceData in their own code, but
should consider replacing the LogisticRegressionInstance with a class more appropriate for the model they construct.

This code is written to be compatible with both Python 2 or 3, at the expense of dependency on the future library. This
code does not depend on any other Python libraries besides future.