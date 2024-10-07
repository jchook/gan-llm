# Essays

The idea is to write a classifier that can tell if a student's essay was written by AI.

## Essay data sources

- https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data
- https://www.kaggle.com/competitions/asap-aes/data

To download these, you must first sign-up for kaggle.com, agree to the rules on each competition and run the command to download the competition data.

```sh
cd datasets
kaggle competitions download -c asap-aes
kaggle competitions download -c learning-agency-lab-automated-essay-scoring-2
dtrx *.zip

# Note we use dtrx because it will unzip them into a folder named after the zip file.
# So the path will look like this:
# datasets/
# ├── asap-aes/
# │   ├── Essay_Set_Descriptions.zip
# │   ├── test_set.tsv
# │   ├── Training_Materials.zip
# │   ├── training_set_rel3.tsv
# │   ├── training_set_rel3.xls
# │   ├── training_set_rel3.xlsx
# │   ├── valid_sample_submission_1_column.csv
# │   ├── valid_sample_submission_1_column_no_header.csv
# │   ├── valid_sample_submission_2_column.csv
# │   ├── valid_sample_submission_5_column.csv
# │   ├── valid_set.tsv
# │   ├── valid_set.xls
# │   └── valid_set.xlsx
# └── learning-agency-lab-automated-essay-scoring-2/
#     ├── learning-agency-lab-automated-essay-scoring-2.zip
#     ├── README.md
#     ├── sample_submission.csv
#     ├── test.csv
#     └── train.csv
```

Other potential sources:

- https://thewritesource.com/studentmodels/
- https://www.essaybank.com/


## Cleaning the data

The two datasets are in different formats and have some oddities about them.

For one, privacy preserving concerns replaced a lot of named entities with @PERSON1, @LOCATION2, @MONEY1, etc. So we have some data cleaning to do.

