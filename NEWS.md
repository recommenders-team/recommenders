<!--
Copyright (c) Recommenders contributors.
Licensed under the MIT License.
-->

# What's New

## Update April 12, 2025

We reached 20,000 stars!!

## Update December 23, 2024

We have a new release [Recommenders 1.2.1](https://github.com/recommenders-team/recommenders/releases/tag/1.2.1)!

We fixed a lot of bugs due to dependencies, improved security, reviewed the notebooks and the libraries.

## Update May 2, 2024

We have a new release [Recommenders 1.2.0](https://github.com/microsoft/recommenders/releases/tag/1.2.0)!

So many changes since our last release. We have full tests on Python 3.8 to 3.11 (around 1800 tests), upgraded performance in many algorithms, reviewed notebooks, and many more improvements.

## Update October 10, 2023

We are pleased to announce that this repository (formerly known as Microsoft Recommenders, https://github.com/microsoft/recommenders), has joined the [Linux Foundation of AI and Data](https://lfaidata.foundation/) (LF AI & Data)! The new organization, `recommenders-team`, reflects this change.

We hope this move makes it easy for anyone to contribute! Our objective continues to be building an ecosystem and a community to sustain open source innovations and collaborations in recommendation systems. 

Now to access the repo, instead of going to https://github.com/microsoft/recommenders, you need to go to https://github.com/recommenders-team/recommenders. The old URL will still resolve to the new one, but we recommend that you update your bookmarks.

## Update August 18, 2023

We moved to a new organization! Now to access the repo, instead of going to https://github.com/microsoft/recommenders, you need to go to https://github.com/recommenders-team/recommenders. The old URL will still resolve to the new one, but we recommend you to update your bookmarks.

## Update February 7, 2023 

We reached 15,000 stars!!

## Update July 20, 2022

We have a new release [Recommenders 1.1.1](https://github.com/microsoft/recommenders/releases/tag/1.1.1)! 

We have introduced a new way of testing our repository using [AzureML](https://azure.microsoft.com/en-us/services/machine-learning/). With AzureML we are able to distribute our tests to different machines and run them in parallel. This allows us to test our repository on a wider range of machines and provides us with a much faster test cycle. Our total computation time went from around 9h to 35min, and we were able to reduce the costs by half. See more details [here](tests/README.md).

We also made other improvements like faster evaluation metrics and improving SAR algorithm. 

## Update April 1, 2022

We have a new release [Recommenders 1.1.0](https://github.com/microsoft/recommenders/releases/tag/1.1.0)! 
We have introduced the SASRec and SSEPT algorithms that are based on transformers. 
In addition, we now have enabled Python 3.8 and 3.9.
We have also made improvements on the SARPlus algorithm, including support for Azure Synapse and Spark 3.2.
There are also bug fixes and improvements on NCF, RBM, LightGBM, LightFM, Scikit-Surprise, the stratified splitter, dockerfile 
and upgrade to Scikit-Learn 1.0.2.

## Update January 13, 2022

We have a new release [Recommenders 1.0.0](https://github.com/microsoft/recommenders/releases/tag/1.0.0)! The codebase has now migrated to TensorFlow versions 2.6 / 2.7 and to Spark version 3. In addition, there are a few changes in the dependencies and extras installed by `pip` (see [this guide](recommenders/README.md#optional-dependencies)). We have also made improvements in the code and the CI / CD pipelines.

## Update September 27, 2021

We have a new release [Recommenders 0.7.0](https://github.com/microsoft/recommenders/releases/tag/0.7.0)!

In this, we have changed the names of the folders which contain the source code, so that they are more informative. This implies that you will need to change any import statements that reference the recommenders package. Specifically, the folder `reco_utils` has been renamed to `recommenders` and its subfolders have been renamed according to [issue 1390](https://github.com/microsoft/recommenders/issues/1390).  

The recommenders package now supports three types of environments: [venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/index.html#) and [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html?highlight=environment#conda-environment) with Python versions 3.6 and 3.7.

We have also added new evaluation metrics: _novelty, serendipity, diversity and coverage_ (see the [evalution notebooks](examples/03_evaluate/README.md)).

Code coverage reports are now generated for every PR, using [Codecov](https://about.codecov.io/).

## Update June 21, 2021

We have a new release [Recommenders 0.6.0](https://github.com/microsoft/recommenders/releases/tag/0.6.0)!

Recommenders is now on PyPI and can be installed using pip! In addition there are lots of bug fixes and utilities improvements.

Here you can find the PyPi page: https://pypi.org/project/recommenders/

Here you can find the package documentation: https://microsoft-recommenders.readthedocs.io/en/latest/

## Update June 1, 2021

We have surpassed 10k stars!

Microsoft Recommenders repository has reached 10k stars and has become the most starred open-source recommender system project on GitHub.

Many thanks and congratulations to all the contributors to this repository! More advanced algorithms and best practices are yet to come!

## Update February 4, 2021

We have a new release [Recommenders 0.5.0](https://github.com/microsoft/recommenders/releases/tag/0.5.0)!

It comes with lots of bug fixes, optimizations and 3 new algorithms, GeoIMC, Standard VAE and Multinomial VAE. We also added tools to facilitate the use of Microsoft News dataset (MIND). In addition, we published our KDD2020 tutorial where we built a recommender of COVID papers using Microsoft Academic Graph.

We also changed the default branch from master to main. Now when you download the repo, you will get main branch.

## Update October 19, 2020

Leaderboard Reopen!

[Microsoft News Recommendation Competition Winners Announced](https://msnews.github.io/competition.html)

Congratulations to all participants and [winners](https://msnews.github.io/competition.html#winner) of the Microsoft News Recommendation Competition!  In the last two months, over 200 participants from more than 90 institutions in 19 countries and regions joined the competition and collectively advanced the state of the art of news recommendation.

The competition is based on the recently released [MIND dataset](https://msnews.github.io/), an open, large-scale English news dataset with impression logs.  Details of the dataset are available at this [ACL paper](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf).

With the competition successfully closed, the [leaderboard](https://msnews.github.io/competition.html#leaderboard) is now reopn.  Want to see if you can grab the top spot? Get familiar with the [news recommendation scenario](https://github.com/microsoft/recommenders/tree/main/scenarios/news).  Then dive into some baselines such as [DKN](examples/00_quick_start/dkn_MIND.ipynb), [LSTUR](examples/00_quick_start/lstur_MIND.ipynb), [NAML](examples/00_quick_start/naml_MIND.ipynb), [NPA](examples/00_quick_start/npa_MIND.ipynb) and [NRMS](examples/00_quick_start/nrms_MIND.ipynb) and start hacking!

## Update October 5, 2020

[Microsoft News Recommendation Competition Winners Announced, Leaderboard to Reopen!](https://msnews.github.io/competition.html)

Congratulations to all participants and [winners](https://msnews.github.io/competition.html#winner) of the Microsoft News Recommendation Competition!  In the last two months, over 200 participants from more than 90 institutions in 19 countries and regions joined the competition and collectively advanced the state of the art of news recommendation.

The competition is based on the recently released [MIND dataset](https://msnews.github.io/), an open, large-scale English news dataset with impression logs.  Details of the dataset are available at this [ACL paper](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf).

With the competition successfully closed, the [leaderboard](https://msnews.github.io/competition.html#leaderboard) will reopen soon.  Want to see if you can grab the top spot? Get familiar with the [news recommendation scenario](https://github.com/microsoft/recommenders/tree/main/scenarios/news).  Then dive into some baselines such as [DKN](examples/00_quick_start/dkn_MIND.ipynb), [LSTUR](examples/00_quick_start/lstur_MIND.ipynb), [NAML](examples/00_quick_start/naml_MIND.ipynb), [NPA](examples/00_quick_start/npa_MIND.ipynb) and [NRMS](examples/00_quick_start/nrms_MIND.ipynb) and get ready!

## Update July 20, 2020

Microsoft is hosting a News Recommendation competition based on the [MIND dataset](https://msnews.github.io/), a large-scale English news dataset with impression logs. Check out the [ACL paper](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf), get familiar with the [news recommendation scenario](https://github.com/microsoft/recommenders/tree/main/scenarios/news), and dive into the [quick start example](examples/00_quick_start/dkn_MIND.ipynb) using the DKN algorithm. Then try some other algorithms (NAML, NPA, NRMS, LSTUR) and tools in recommenders and submit your entry!

## Update August 20, 2020

New release: [Recommenders 0.4.0](https://github.com/microsoft/recommenders/releases/tag/0.4.0)

13 new algos and multiple fixes and new features

## Update September 18, 2019

New release: [Recommenders 0.3.1](https://github.com/microsoft/recommenders/releases/tag/0.3.1)

## Update September 15, 2019

We reached 5000 stars!!

## Update June 3, 2019

New release: [Recommenders 0.3.0](https://github.com/microsoft/recommenders/releases/tag/0.3.0)

## Update February 20, 2019

New release: [Recommenders 0.2.0](https://github.com/microsoft/recommenders/releases/tag/0.2.0)

## Update February 11, 2019

We reached 1000 stars!!

## Update December 12, 2018

First release: [Recommenders 0.1.1](https://github.com/microsoft/recommenders/releases/tag/0.1.1)

## Update November 12, 2018

First pre-release: [Recommenders 0.1.0](https://github.com/microsoft/recommenders/releases/tag/0.1.0)
