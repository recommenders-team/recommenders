<!--
Copyright (c) Recommenders contributors.
Licensed under the MIT License.
-->

# Contribution Guidelines

Contributions are welcomed! Here's a few things to know:

- [Steps to Contributing](#steps-to-contributing)
- [Ideas for Contributions](#ideas-for-contributions)
  - [A first contribution](#a-first-contribution)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Metrics](#metrics)
  - [General tips](#general-tips)
- [Coding Guidelines](#coding-guidelines)
- [Code of Conduct](#code-of-conduct)
  - [Do not point fingers](#do-not-point-fingers)
  - [Provide code feedback based on evidence](#provide-code-feedback-based-on-evidence)
  - [Ask questions do not give answers](#ask-questions-do-not-give-answers)

## Steps to Contributing

**TL;DR for contributing: We use the staging branch to land all new features and fixes. To make a contribution, please create a branch from staging, make a modification in the code and create a PR to staging.** 

Here are the basic steps to get started with your first contribution. Please reach out with any questions.
1. Use [open issues](https://github.com/Microsoft/Recommenders/issues) to discuss the proposed changes. Create an issue describing changes if necessary to collect feedback. Also, please use provided labels to tag issues so everyone can easily sort issues of interest.
1. [Fork the repo](https://help.github.com/articles/fork-a-repo/) so you can make and test local changes.
1. Create a new branch **from staging branch** for the issue (please do not create a branch from main). We suggest prefixing the branch with your username and then a descriptive title: (e.g. `gramhagen/update_contributing_docs`)
1. Install recommenders package locally using the right optional dependency for your test and the dev option. (e.g. gpu test: `pip install -e .[gpu,dev]`)
1. Create a test that replicates the issue.
1. Make code changes.
1. Ensure unit tests pass and code style / formatting is consistent (see [wiki](https://github.com/Microsoft/Recommenders/wiki/Coding-Guidelines#python-and-docstrings-style) for more details).
1. When adding code to the repo, make sure you sign the commits, otherwise the tests will fail (see [how to sign the commits](https://github.com/recommenders-team/recommenders/wiki/How-to-sign-commits)).
1. Create a pull request against **staging** branch.

See the wiki for more details about our [merging strategy](https://github.com/microsoft/recommenders/wiki/Strategy-to-merge-the-code-to-main-branch).

## Ideas for Contributions

### A first contribution

For people who are new to open source or to Recommenders, a good way to start is by contribution with documentation. You can help with any of the README files or in the notebooks.

For more advanced users, consider fixing one of the bugs listed in the issues.

### Datasets

To contribute new datasets, please consider this:

* Minimize dependencies, it's better to use `requests` library than a custom library.
* Make sure that the dataset is publicly available and that the license allows for redistribution.

### Models

To contribute new models, please consider this:

* Please don't add models that are already implemented in the repo. An exception to this rule is if you are adding a more optimal implementation or you want to migrate a model from TensorFlow to PyTorch.
* Prioritize the minimal code necessary instead of adding a full library. If you add code from another repository, please make sure to follow the license and give proper credit.
* All models should be accompanied by a notebook that shows how to use the model and how to train it. The notebook should be in the [examples](examples) folder.
* The model should be tested with unit tests, and the notebooks should be tested with functional tests.

### Metrics

To contribute new metrics, please consider this:

* A good way to contribute with metrics is by optimizing the code of the existing ones.
* If you are adding a new metric, please consider adding not only a CPU version, but also a PySpark version.
* When adding the tests, make sure you check for the limits. For example, if you add an error metric, check that the error between two identical datasets is zero.

### General tips

* Prioritize PyTorch over TensorFlow.
* Minimize dependencies. Around 80% of the issues in the repo are related to dependencies.
* Avoid adding code with GPL and other copyleft licenses. Prioritize MIT, Apache, and other permissive licenses.
* Add the copyright statement at the beginning of the file: `Copyright (c) Recommenders contributors. Licensed under the MIT License.`

## Coding Guidelines

We strive to maintain high quality code to make the utilities in the repository easy to understand, use, and extend. We also work hard to maintain a friendly and constructive environment. We've found that having clear expectations on the development process and consistent style helps to ensure everyone can contribute and collaborate effectively.

Please review the [Coding Guidelines](https://github.com/recommenders-team/recommenders/wiki/Coding-Guidelines) wiki page to see more details about the expectations for development approach and style.

## Code of Conduct

Apart from the official [Code of Conduct](CODE_OF_CONDUCT.md), in Recommenders team we adopt the following behaviors, to ensure a great working environment:

### Do not point fingers
Let’s be constructive.

<details>
<summary><em>Click here to see some examples</em></summary>

"This method is missing docstrings" instead of "YOU forgot to put docstrings".

</details>

### Provide code feedback based on evidence 

When making code reviews, try to support your ideas based on evidence (papers, library documentation, stackoverflow, etc) rather than your personal preferences. 

<details>
<summary><em>Click here to see some examples</em></summary>

"When reviewing this code, I saw that the Python implementation of the metrics are based on classes, however, [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) use functions. We should follow the standard in the industry."

</details>

### Ask questions do not give answers
Try to be empathic. 

<details>
<summary><em>Click here to see some examples</em></summary>

* Would it make more sense if ...?
* Have you considered this ... ?

</details>

