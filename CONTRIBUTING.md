# Contribution Guidelines

Here you will find the contribution guidelines.


## Microsoft CLA

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Recommenders Team Contribution Guidelines

### Practices Related to Coding

#### Test Driven Development (TDD) 

All contributions to the repository should have unit tests, we use [pytest](https://docs.pytest.org/en/latest/) for Python files and [papermill](https://github.com/nteract/papermill) for notebooks. 

Apart from unit tests, we also have nightly builds with smoke and integration tests. For more information, see a [quick introduction to unit, smoke and integration tests](https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/).

*Examples:*

* Basic asserts with [fixtures comparing structures like list, dictionaries, numpy arrays and pandas dataframes](https://github.com/miguelgfierro/codebase/blob/master/python/test/pytest_fixtures.py).
* Basic use of [common fixtures defined in a conftest file](https://github.com/miguelgfierro/codebase/blob/master/python/test/pytest_fixtures_in_common_file.py).
* Python unit tests for our [evaluation metrics](tests/unit/test_python_evaluation.py).
* Notebook unit tests for our [PySpark notebooks](tests/unit/test_notebooks_pyspark.py).

#### Don’t Repeat Yourself (DRY)

DRY by refactoring common code.

*Examples:*

* See how we are using [DRY when testing our notebooks](tests/notebooks_common.py). 

#### Single Responsibility

One of the [SOLID](https://en.wikipedia.org/wiki/SOLID) principles, it states that each module or function should have responsibility over a single part of the functionality. 

*Examples:*

Without single responsibility:
```
def train_and_test(train_set, test_set):
    # code for training on train set
    # code for testing on test_set
```
With single responsibility:
```
def train(train_set):
    # code for training on train set

def test(test_set):
    # code for testing on test_set  
```

#### Python and Docstrings Style
We use the automatic style formatter [Black](https://github.com/ambv/black). 

We use [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting the docstrings.

*Examples:*

* [Black formatting on Python files](https://github.com/ambv/black#the-black-code-style). 
* [Black formatting on Notebooks](https://github.com/csurfer/blackcellmagic).
* [Docstring with Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

#### The Zen of Python
We follow the [Zen of Python](https://www.python.org/dev/peps/pep-0020/).

```
Beautiful is better than ugly.
Explicit is better than implicit. 
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those! 
```

*Examples:*

Implementation of [explicit is better than implicit](https://miguelgfierro.com/blog/2018/python-pro-tips-understanding-explicit-is-better-than-implicit/) with a read function:
```
#Implicit
def read(filename):
    # code for reading a csv or json
    # depending on the file extension

#Explicit
def read_csv(filename):
    # code for reading a csv

def read_json(filename):
    # code for reading a json

```

### Practices Related to Software Design and Decision Making

#### Evidence-Based Software Design (EBD)
Software is developed based on customer inputs, standard libraries in the industry or credible research. For a detailed explanation, see this [post about EBD](https://miguelgfierro.com/blog/2018/evidence-based-software-design/). 

*Examples:*

When designing the interfaces of the evaluation metrics in Python, we took the decision of using functions instead of classes, following standards in the industry like [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) and [tensorflow](https://www.tensorflow.org/api_docs/python/tf/metrics). See our implementation of [Python metrics](reco_utils/evaluation/python_evaluation.py).

#### You aren’t going to need it (YAGNI)

We should only implement functionalities when we need them and not when we foresee we might need them.

*Examples:*

* Question: should we start developing now computer vision capabilities for when our boss buys us the [SpotMini robot](https://www.youtube.com/watch?v=kHBcVlqpvZ8)?
* Answer: No, SpotMini is not for sale yet and we still need to convince our boss to buy it.

### Practices related to project management

#### Minimum Viable Product (MVP)

We work through MVPs, which are our milestones. An MVP is that version of a new product which allows a team to collect the maximum amount of validated learning about customers with the least effort. More information about MVPs can be found in the [Lean Startup methodology](http://theleanstartup.com/principles).

*Examples:*

* [Initial MVP of our repo](https://github.com/Microsoft/Recommenders/milestone/1) with basic functionality.
* [Second MVP to give early access](https://github.com/Microsoft/Recommenders/milestone/3) to selected users and customers.

#### Publish Often Publish Early (PEPO)
Even before we have an MVP, get the code base working and doing something, even if it is something trivial that everyone can "run" easily. 

*Examples:*

We make sure that in between MVPs all the code that goes to the branches staging or master passes the tests.


### Practices related to customer focus

#### Get customer feedback before making a release
A product cycle is not finished until we get feedback from a customer/user, we have made changes based on the feedback and all the tests are passing.

*Examples:*

* See our [branch merging strategy](https://github.com/Microsoft/Recommenders/wiki/Strategy-to-merge-the-code-to-master-branch).

### Practices related to team behavior

#### Don’t point fingers
Let’s be constructive.

*Examples:*

"This method is missing docstrings" instead of "YOU forgot to put docstrings".

#### Provide code feedback based on evidence 

When giving feedback about a piece of code, try to support your ideas based on evidence (papers, library documentation, stackoverflow, etc) rather than your personal preferences. 

*Examples:*

* When reviewing this code, I saw that the Python implementation the metrics are based on classes, however, [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) and [tensorflow](https://www.tensorflow.org/api_docs/python/tf/metrics) use functions. We should follow the standard in the industry.

#### Ask questions, don’t give answers
Try to be empathic. 

*Examples:*

* Would it make more sense if ...?
* Have you considered this ... ?


