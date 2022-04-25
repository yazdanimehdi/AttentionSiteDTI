# AttentionSiteDTI
This is The repository for supporting matterial of "Interpretable and Generalizable Attention-Based Model for Predicting Drug-Target Interaction Using 3D Structure of Protein Binding Sites: SARS-CoV-2 Case Study and in-Lab Validation"
https://www.biorxiv.org/content/10.1101/2021.12.07.471693v2

![AttentionSiteDTI](AttentionSiteDTI.png)

## Requirements
A suitable [conda](https://conda.io/) environment can be created:
```
conda env create -f noveldti.yml
conda activate noveldti
```
## Data
All data used in this paper are publicly available and can be accessed here: [DUD-E](http://dude.docking.org ), [BindingDB-IBM dataset](https://github.com/IBM/InterpretableDTIP), [Human dataset](https://github.com/masashitsubaki/CPI_prediction/tree/master/dataset) and [human sequence to pdb](https://github.com/prokia/drugVQA/tree/master/data)

## Demo Instructions
After downloading the human dataset you and place it in the project root folder you can generate the preprocessed data by running
```
python human_data.py
```
After generating the human_part_train.pkl, human_part_val.pkl and human_part_test.pkl you can start training the model by running
```
python main2.py
```




