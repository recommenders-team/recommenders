# Data Preparation

In this directory, notebooks are provided to illustrate [utility functions](../../reco_utils) for
data operations such as data import / export, data transformation, data split, etc., which are frequent
data preparation tasks witnessed in recommendation system development.

| Notebook | Description | 
| --- | --- | 
| [data_split](data_split.ipynb) | Details on splitting data (randomly, chronologically, etc). |
| [data_transform](data_transform.ipynb) | Guidance on how to transform (implicit / explicit) data for building collaborative filtering typed recommender. |
| [wikidata knowledge graph](wikidata_KG.ipynb) | Details on how to create a knowledge graph using Wikidata |

### Data split

Three methods of splitting the data for training and testing are demonstrated in this notebook. Each supports both Spark and pandas DataFrames.
1. Random Split: this is the simplest way to split the data, it randomly assigns entries to either the train set or the test set based on the allocation ratio desired.
2. Chronological Split: in many cases accounting for temporal variations when evaluating your model can provide more realistic measures of performance. This approach will split the train and test set based on timestamps by user or item.
3. Stratified Split: it may be preferable to ensure the same set of users or items are in the train and test sets, this method of splitting will ensure that is the case.

###  Data transform
Data transformation techniques which are commonly used in various recommendation scenarios are introduced and reviewed. 
