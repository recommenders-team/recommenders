# Glossary

* A/B testing: Methodology to evaluate the performance of a system in production. In the context of Recommendation Systems it is used to measure a machine learning model performance in real-time. It works by randomizing an environment response into two groups A and B, typically half of the traffic goes to the machine learning model output and the other half is left without model. By comparing the metrics from A and B branches, it is possible to evaluate whether it is beneficial the use of the model or not. A test with more than two groups it is named Multi-Variate Test.

* Click-through rate (CTR): Ratio of the number of users who click on a link over the total number of users that visited the page. CTR is a measure of the user engagement.

* Cold-start problem: The cold start problem concerns the recommendations for users with no or few past history (new users). Providing recommendations to users with small past history becomes a difficult problem for collaborative filtering models because their learning and predictive ability is limited. Multiple research have been conducted in this direction using content-based filtering models or hybrid models. These models use auxiliary information like user or item metadata to overcome the cold start problem.

* Collaborative filtering algorithms (CF): CF algorithms make prediction of what is the likelihood of a user selecting an item based on the behavior of other users [1]. It assumes that if user A likes item X and Y, and user B likes item X, user B would probably like item Y. See the [list of CF examples in Recommenders repository](../../examples/02_model_collaborative_filtering).

* Content-based filtering algorithms (CB): CB algorithms make prediction of what is the likelihood of a user selecting an item based on the similarity of users and items among themselves [1]. It assumes that if user A lives in country X, has age Y and likes item Z, and user B lives in country X and has age Y, user B would probably like item Z. See the [list of CB examples in Recommenders repository](../../examples/02_model_content_based_filtering).

* Conversion rate: In the context of e-commerce, the conversion rate is the ratio between the number of conversions (e.g. number of bought items) over the total number of visits. In the context of recommendation systems, conversion rate measures how efficient is an algorithm to provide recommendations that the user buys.

* Diversity metrics: In the context of Recommendation Systems,  diversity applies to a set of items, and is related to how different the items are with respect to each other [4].

* Explicit interaction data: When a user explicitly rate an item, typically between 1-5, the user is giving a value on the likeliness of the item. 

* Hybrid filtering algorithms: This type of recommendation system can implement a combination of collaborative and content-based filtering models. See the [list of examples in Recommenders repository](../../examples/02_model_hybrid).

* Implicit interaction data: Implicit interactions are views or clicks that show a certain interest of the user about a specific items. These kind of data is more common but it doesn't define the intention of the user as clearly as the explicit data.

* Item information: These include information about the item, some examples can be name, description, price, etc.

* Knowledge-base algorithms: ...

* Knowledge graph data: ...

* Long tail products: Typically, the shape of items interacted in retail follow a long tail distribution [1,2]....

* Multi-Variate Test (MVT): Methodology to evaluate the performance of a system in production. It is similar to A/B testing, with the difference that instead of having two test groups, MVT has multiples groups. 

* Novelty metrics: In Recommendation Systems, the novelty of a piece of information generally refers to how different it is with respect to "what has been previously seen" [4]. 

* Online metrics: 

* Offline metrics: Metrics computed offline for measuring the performance of the machine learning model. These metrics include ranking, rating, diversity and novelty metrics.

* Ranking metrics:

* Rating metrics:

* Revenue per order: The revenue per order optimization objective is the default optimization objective for the "Frequently bought together" recommendation model type. This optimization objective cannot be specified for any other recommendation model type.

* User information: These include all information that define the user, some examples can be name, address, email, demographics, etc. 

## References and resources

[1] Aggarwal, Charu C. Recommender systems. Vol. 1. Cham: Springer International Publishing, 2016.
[2]. Park, Yoon-Joo, and Tuzhilin, Alexander. "The long tail of recommender systems and how to leverage it." In Proceedings of the 2008 ACM conference on Recommender systems, pp. 11-18. 2008. [Link to paper](http://people.stern.nyu.edu/atuzhili/pdf/Park-Tuzhilin-RecSys08-final.pdf).
[3]. Armstrong, Robert. "The long tail: Why the future of business is selling less of more." Canadian Journal of Communication 33, no. 1 (2008). [Link to paper](https://www.cjc-online.ca/index.php/journal/article/view/1946/3141).
[4] Castells, P., Vargas, S., and Wang, Jun. "Novelty and diversity metrics for recommender systems: choice, discovery and relevance." (2011). [Link to paper](https://repositorio.uam.es/bitstream/handle/10486/666094/novelty_castells_DDR_2011.pdf?sequence=1).






