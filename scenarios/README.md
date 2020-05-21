# Recommendation System Scenarios

On this section there is listed a number of business scenarios that are common in Recommendation Systems (RS).

The list of scenarios are:

* Ads
* Entertainment
* Food and restaurants
* News
* Retail
* Travel

## Types of Recommendation Systems

Typically recommendation systems in retail can be divided into three categories:

* Collaborative filtering: This type of recommendation system makes predictions of what might interest a person based on the taste of many other users. It assumes that if person X likes Snickers, and person Y likes Snickers and Milky Way, then person X might like Milky Way as well.

* Content-based filtering: This type of recommendation system focuses on the products themselves and recommends other products that have similar attributes. Content-based filtering relies on the characteristics of the products themselves, so it doesn’t rely on other users to interact with the products before making a recommendation.

* Hybrid filtering: This type of recommendation system can implement a combination fo any two of the above systems.

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



