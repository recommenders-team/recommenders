# Recommendation System Scenarios

On this section there is listed a number of business scenarios that are common in Recommendation Systems.

The list of scenarios are:

* [Ads](ads)
* [Entertainment](entertainment)
* [Food and restaurants](food_and_restaurants)
* [News and document]()
* [Retail](retail)
* [Travel](travel)


## Data in Recommendation Systems

### Data types

* Explicit interactions:

* Implicit interactions:

* Knowledge graph data:

* User features:

* Item features:

### Considerations about data size

The size of the data is important when designing the system...


## Metrics

In RS, there are two types of metrics: offline and online metrics.

### Machine learning metrics (offline metrics)

In Recommenders, offine metrics implementation for python are found on [python_evaluation.py](https://github.com/microsoft/recommenders/blob/master/reco_utils/evaluation/python_evaluation.py) and those for PySpark are found on [spark_evaluation.py](https://github.com/microsoft/recommenders/blob/master/reco_utils/evaluation/spark_evaluation.py).

Currently available metrics include:

- Root Mean Squared Error
- Mean Absolute Error
- R<sup>2</sup>
- Explained Variance
- Precision at K
- Recall at K
- Normalized Discounted Cumulative Gain at K
- Mean Average Precision at K
- Area Under Curve
- Logistic Loss


### Business success metrics (online metrics)

Online metrics are specific on the business scenario. More details can be found on each scenario folder.

## Managing Cold Start Scenarios in Recommendation Systems

....



