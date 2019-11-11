# 11785-project	

### [Second Language Acuisition Modeling](docs/SLAM.pdf)
Given a history of token-level errors made by the learner in the learning language (L2), accurately predict the errors they will make in the future. 

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

### Run the script
#### es_en
```bash
cd src
python3 main.py --train ../data/es_en/es_en.slam.20190204.train --test ../data/es_en/es_en.slam.20190204.test --pred ./es_en_predictions.pred
python3 ../baseline/eval.py --pred ./es_en_predictions.pred --key ../data/es_en/es_en.slam.20190204.test.key
```
#### fr_en
```bash
cd src
python3 main.py --train ../data/fr_en/fr_en.slam.20190204.train --test ../data/fr_en/fr_en.slam.20190204.test --pred ./fr_en_predictions.pred
python3 ../baseline/eval.py --pred ./fr_en_predictions.pred --key ../data/fr_en/fr_en.slam.20190204.test.key
```
#### en_es
```bash
cd src
python3 main.py --train ../data/en_es/en_es.slam.20190204.train --test ../data/en_es/en_es.slam.20190204.test --pred ./en_es_predictions.pred
python3 ../baseline/eval.py --pred ./en_es_predictions.pred --key ../data/en_es/en_es.slam.20190204.test.key
```

### Class Documents	
[Project Proposal Guidelines](docs/Project_Proposal_Guidelines.pdf)	


### Datasets	
[Duolingo AI](https://ai.duolingo.com)	


### [Related Works](docs/publications.md)

### Results
#### es_en
| model               | accuracy | avglogloss | auroc | F1    |
| ---                 | ---      | ---        | ---   | ---   |
| Random              | 0.500    | 0.998      | 0.500 | 0.243 |
| Logistic Regression | 0.844    | 0.386      | 0.745 | 0.183 |
| Perceptron          | 0.839    | 4.712      | 0.538 | 0.000 |