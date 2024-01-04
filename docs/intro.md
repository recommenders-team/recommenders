<!--
Copyright (c) Recommenders contributors.
Licensed under the MIT License.
-->

# Welcome to Recommenders

Recommenders objective is to assist researchers, developers and enthusiasts in prototyping, experimenting with and bringing to production a range of classic and state-of-the-art recommendation systems.

````{margin}
```sh
pip install recommenders
```
<a class="github-button" href="https://github.com/recommenders-team/recommenders" data-icon="octicon-star" style="margin:auto" data-size="large" data-show-count="true" aria-label="Star Recommenders on GitHub">Star Us</a><script async defer src="https://buttons.github.io/buttons.js"></script>
````

Recommenders is a project under the [Linux Foundation of AI and Data](https://lfaidata.foundation/projects/). 

This repository contains examples and best practices for building recommendation systems, provided as Jupyter notebooks.

The examples detail our learnings on five key tasks:

- Prepare Data: Preparing and loading data for each recommendation algorithm.
- Model: Building models using various classical and deep learning recommendation algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) or eXtreme Deep Factorization Machines ([xDeepFM](https://arxiv.org/abs/1803.05170)).
- Evaluate: Evaluating algorithms with offline metrics.
- Model Select and Optimize: Tuning and optimizing hyperparameters for recommendation models.
- Operationalize: Operationalizing models in a production environment.

Several utilities are provided in the `recommenders` library to support common tasks such as loading datasets in the format expected by different algorithms, evaluating model outputs, and splitting training/test data. Implementations of several state-of-the-art algorithms are included for self-study and customization in your own applications.


<!-- ```{tableofcontents}
``` -->

