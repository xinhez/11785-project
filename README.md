# 11785-project	Second Language Acquisition Modeling 
[Read the Duolingo overview paper](docs/papers/SLAM.pdf)

Given a history of token-level errors made by the learner in the learning language (L2), accurately predict the errors they will make in the future. 

--- 

### Group Members:
- [Xinhe Zhang](xinhez@andrew.cmu.edu)
- [Xinyue Zhang](xzhang4@andrew.cmu.edu)
- [Xueqian Zhang](xueqianz@andrew.cmu.edu)

--- 

### Our Results

|       | es_en    |            |       |       |
| ---   | ---      | ---        | ---   | ---   |
| model | accuracy | avglogloss | auroc | F1    |
| [Random](src/models/Random.py) | 0.500 | 0.998 | 0.500 | 0.243 |
| [Logistic Regression](src/models/LogisticRegression.py) | 0.844 | 0.386 | 0.745 | 0.183 |
| [Perceptron](src/models/Perceptron.py) | 0.839 | 0.437 | 0.608 | 0.000 |
| [Seq2seq](src/models/Seq2seq.py) | 0.857 | 0.427 | 0.667 | 0.015 |
| [Seq2seq_Exp](src/models/Seq2seq_Exp.py) | 0.842 | 0.400 | 0.735 | 0.267 |

#### Seq2seq
| Epoch | accuracy | avglogloss | auroc | F1    |
| ---   | ---      | ---        | ---   | ---   |
| 1     | 0.842    | 0.412      | 0.680 | 0.069 |
| 2     | 0.843    | 0.408      | 0.693 | 0.103 |
| 3     | 0.843    | 0.405      | 0.696 | 0.111 | 
| 4     | 0.843    | 0.406      | 0.697 | 0.102 | <-
| 5     | 0.843    | 0.406      | 0.692 | 0.118 |
| 6     | 0.843    | 0.408      | 0.690 | 0.080 | 
| 7     | 0.842    | 0.418      | 0.686 | 0.112 |
| 8     | 0.842    | 0.414      | 0.694 | 0.133 |
| 9     | 0.840    | 0.425      | 0.682 | 0.156 |
| 10    | 0.817    | 0.455      | 0.688 | 0.283 |

#### Seq2seq with MLP Grader
| Epoch | accuracy | avglogloss | auroc | F1    |
| ---   | ---      | ---        | ---   | ---   |
| 1     | 0.843    | 0.408      | 0.689 | 0.082 |
| 2     | 0.843    | 0.407      | 0.695 | 0.104 |
| 3     | 0.842    | 0.410      | 0.694 | 0.131 |
| 4     | 0.843    | 0.409      | 0.693 | 0.065 |
| 5     | 0.843    | 0.405      | 0.699 | 0.092 | <-
| 6     | 0.842    | 0.408      | 0.693 | 0.129 |
| 7     | 0.841    | 0.418      | 0.671 | 0.093 |
| 8     | 0.841    | 0.412      | 0.679 | 0.036 |
| 9     | 0.840    | 0.413      | 0.685 | 0.132 |
| 10    | 0.838    | 0.427      | 0.645 | 0.092 |

#### Seq2seq with MLP Grader and User ID
| Epoch | accuracy | avglogloss | auroc | F1    |
| ---   | ---      | ---        | ---   | ---   |
| 1     | 0.827    | 0.439      | 0.650 | 0.162 |
| 2     | 0.834    | 0.435      | 0.653 | 0.138 |
| 3     | 0.834    | 0.429      | 0.661 | 0.165 |
| 4     | 0.821    | 0.441      | 0.677 | 0.236 | <-
| 5     | 0.837    | 0.424      | 0.658 | 0.130 | 
| 6     | 0.832    | 0.431      | 0.653 | 0.143 | 
| 7     | 0.831    | 0.439      | 0.658 | 0.176 | 
| 8     | 0.835    | 0.427      | 0.658 | 0.155 | 
| 9     | 0.828    | 0.442      | 0.640 | 0.164 |
| 10    | 0.833    | 0.438      | 0.654 | 0.159 |


#### Attention (torchnlp) based on Seq2seq
| Epoch | accuracy | avglogloss | auroc | F1    |
| ---   | ---      | ---        | ---   | ---   |
| 1     | 0.651    | 0.751      | 0.552 | 0.240 |
| 2     | 0.788    | 0.538      | 0.521 | 0.111 |

#### CNN
| Epoch | accuracy | avglogloss | auroc | F1    |
| ---   | ---      | ---        | ---   | ---   |
| 1     | 0.837    | 0.425      | 0.654 | 0.094 |
| 2     | 0.830    | 0.439      | 0.636 | 0.117 |
| 3     | 0.830    | 0.438      | 0.638 | 0.114 |

---

## Example Command
### Install this project

```bash
git clone https://github.com/xinhez/11785-project.git
cd 11785-project/data
mkdir en_es
tar -xvzf data_en_es.tar.gz --directory en_es
mkdir es_en
tar -xvzf data_es_en.tar.gz --directory es_en
mkdir fr_en
tar -xvzf data_fr_en.tar.gz --directory fr_en
```
---

### Run the script
#### es_en
```bash
cd src

python3 main.py --language es_en

python3 ../baseline/eval.py --pred ./outputs/es_en_dev_predictions.pred --key ../data/es_en/es_en.slam.20190204.dev.key

python3 ../baseline/eval.py --pred ./outputs/es_en_test_predictions.pred --key ../data/es_en/es_en.slam.20190204.test.key
```
#### fr_en
```bash
cd src

python3 main.py --language fr_en

python3 ../baseline/eval.py --pred ./outputs/fr_en_dev_predictions.pred --key ../data/fr_en/fr_en.slam.20190204.dev.key

python3 ../baseline/eval.py --pred ./outputs/fr_en_test_predictions.pred --key ../data/fr_en/fr_en.slam.20190204.test.key
```
#### en_es
```bash
cd src

python3 main.py --language en_es

python3 ../baseline/eval.py --pred ./outputs/en_es_dev_predictions.pred --key ../data/en_es/en_es.slam.20190204.dev.key

python3 ../baseline/eval.py --pred ./outputs/en_es_test_predictions.pred --key ../data/en_es/en_es.slam.20190204.test.key
```
---

### Class Documents	
- [Project Proposal Guidelines](docs/Project_Proposal_Guidelines.pdf)	
- [Read Our Midterm Report](docs/submissions/midterm.pdf)

---

### Related Works
|      | en_es |     | es_en |     | fr_en |     |      |
| ---  | ---   | --- | ---   | --- | ---   | --- | ---  |
| Team | auc   | f1  | auc   | f1  | auc   |  f1 | rank |
| [SanaLabs ♢♣](docs/papers/osika.slam18.pdf) | 0.861 | 0.561 | 0.838 | 0.530 | 0.857 | 0.573 | 1.0 |
| [singsound ♢](docs/papers/xu.slam18.pdf) | 0.861 | 0.561 | 0.835 | 0.524 | 0.854 | 0.569 | 1.7 |
| [NYU ♣‡](docs/papers/rich.slam18.pdf) | 0.859 | 0.468 | 0.835 | 0.420 | 0.854 | 0.493 | 2.3 |
| [TMU ♢‡](docs/papers/kaneko.slam18.pdf) | 0.848 | 0.476 | 0.824 |	0.439 |	0.839 |	0.502 |	4.3 |
| [CECL ‡](docs/papers/bestgen.slam18.pdf) | 0.846 | 0.414 | 0.818 | 0.390 | 0.843 | 0.487 | 4.7 |
| [Cambridge ♢](docs/papers/yuan.slam18.pdf) | 0.841 | 0.479 | 0.807 | 0.435 | 0.835 | 0.508 | 6.0 | 
| [UCSD ♣](docs/papers/tomoschuk.slam18.pdf) | 0.829 | 0.424 | 0.803 | 0.375 | 0.823 | 0.442 | 7.0 | 
| [LambdaLab ♣](docs/papers/chen.slam18.pdf) | 0.821 | 0.389 | 0.801 | 0.344 | 0.815 | 0.415 | 7.6 | 
| [Grotoco](docs/papers/klerke.slam18.pdf) | 0.817 | 0.462 | 0.791 | 0.452 | 0.813 | 0.502 | 9.0 | 
| [nihalnayak](docs/papers/nayak.slam18.pdf) | 0.821 | 0.376 | 0.790 | 0.338 | 0.811 | 0.431 | 9.0 | 
| [jilljenn](docs/papers/vie.slam18.pdf) | 0.815 | 0.329 | 0.788 | 0.306 | 0.809 | 0.406 | 10.7 | 
| SLAM_baseline | 0.774 | 0.190 | 0.746 | 0.175 | 0.771 | 0.281 | 14.7 | 

Annotations:
- ♢: recurrent neural networks 
- ♣: decision tree ensembles
- ‡: multitask framework