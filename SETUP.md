<!--
Copyright (c) Recommenders contributors.
Licensed under the MIT License.
-->

# Setup Guide

The repo, including this guide, is tested on Linux. Where applicable, we document differences in [Windows](#windows-specific-instructions) and [MacOS](#macos-specific-instructions) although 
such documentation may not always be up to date.   

## Extras

In addition to the pip installable package, several extras are provided, including:
+ `[gpu]`: Needed for running GPU models.  
+ `[spark]`: Needed for running Spark models.
+ `[dev]`: Needed for development.
+ `[all]`: `[gpu]`|`[spark]`|`[dev]`
+ `[experimental]`: Models that are not thoroughly tested and/or may require additional steps in installation).

## Setup for Core Package

Follow the [Getting Started](./README.md#Getting-Started) section in the [README](./README.md) to install the package and run the examples.

## Setup for GPU

```bash
# 1. Make sure CUDA is installed.

# 2. Follow Steps 1-5 in the Getting Started section in README.md to install the package and Jupyter kernel, adding the gpu extra to the pip install command:
pip install recommenders[gpu]

# 3. Within VSCode:
#   a. Open a notebook with a GPU model, e.g., examples/00_quick_start/wide_deep_movielens.ipynb;
#   b. Select Jupyter kernel <kernel_name>;
#   c. Run the notebook.
```

## Setup for Spark 

```bash
# 1. Make sure JDK is installed.  For example, OpenJDK 11 can be installed using the command
# sudo apt-get install openjdk-11-jdk

# 2. Follow Steps 1-5 in the Getting Started section in README.md to install the package and Jupyter kernel, adding the spark extra to the pip install command:
pip install recommenders[spark]

# 3. Within VSCode:
#   a. Open a notebook with a Spark model, e.g., examples/00_quick_start/als_movielens.ipynb;  
#   b. Select Jupyter kernel <kernel_name>;
#   c. Run the notebook.
```

## Setup for Databricks

The following instructions were tested on Databricks Runtime 15.4 LTS (Apache Spark version 3.5.0), 14.3 LTS (Apache Spark version 3.5.0), 13.3 LTS (Apache Spark version 3.4.1), and 12.2 LTS (Apache Spark version 3.3.2). We have tested the runtime on python 3.9,3.10 and 3.11. 

After an Databricks cluster is provisioned:
```bash
# 1. Go to the "Compute" tab on the left of the page, click on the provisioned cluster and then click on "Libraries". 
# 2. Click the "Install new" button.  
# 3. In the popup window, select "PyPI" as the library source. Enter "recommenders[examples]" as the package name. Click "Install" to install the package.
# 4. Now, repeat the step 3 for below packages:
#   a. numpy<2.0.0
#   b. pandera<=0.18.3
#   c. scipy<=1.13.1
```

### Prepare Azure Databricks for Operationalization
<!-- TO DO: This is to be verified/updated 23/04/16 -->
This repository includes an end-to-end example notebook that uses Azure Databricks to estimate a recommendation model using matrix factorization with Alternating Least Squares, writes pre-computed recommendations to Azure Cosmos DB, and then creates a real-time scoring service that retrieves the recommendations from Cosmos DB. In order to execute that [notebook](examples/05_operationalize/als_movie_o16n.ipynb), you must install the Recommenders repository as a library (as described above), **AND** you must also install some additional dependencies. With the *Quick install* method, you just need to pass an additional option to the [installation script](tools/databricks_install.py).

<details>
<summary><strong><em>Quick install</em></strong></summary>

This option utilizes the installation script to do the setup. Just run the installation script
with an additional option. If you have already run the script once to upload and install the `Recommenders.egg` library, you can also add an `--overwrite` option:

```{shell}
python tools/databricks_install.py --overwrite --prepare-o16n <CLUSTER_ID>
```

This script does all of the steps described in the *Manual setup* section below.

</details>

<details>
<summary><strong><em>Manual setup</em></strong></summary>

You must install three packages as libraries from PyPI:

* `azure-cli==2.0.56`
* `azureml-sdk[databricks]==1.0.8`
* `pydocumentdb==2.3.3`

You can follow instructions [here](https://docs.azuredatabricks.net/user-guide/libraries.html#install-a-library-on-a-cluster) for details on how to install packages from PyPI.

Additionally, you must install the [spark-cosmosdb connector](https://docs.databricks.com/spark/latest/data-sources/azure/cosmosdb-connector.html) on the cluster. The easiest way to manually do that is to:


1. Download the [appropriate jar](https://search.maven.org/remotecontent?filepath=com/azure/cosmos/spark/azure-cosmos-spark_3-1_2-12/4.3.1/azure-cosmos-spark_3-1_2-12-4.3.1.jar) from MAVEN. **NOTE** This is the appropriate jar for spark versions `3.1.X`, and is the appropriate version for the recommended Azure Databricks run-time detailed above. See the [Databricks installation script](https://github.com/microsoft/recommenders/blob/main/tools/databricks_install.py#L45) for other Databricks runtimes.
2. Upload and install the jar by:
   1. Log into your `Azure Databricks` workspace
   2. Select the `Clusters` button on the left.
   3. Select the cluster on which you want to import the library.
   4. Select the `Upload` and `Jar` options, and click in the box that has the text `Drop JAR here` in it.
   5. Navigate to the downloaded `.jar` file, select it, and click `Open`.
   6. Click on `Install`.
   7. Restart the cluster.

</details>


## Setup for Experimental 
<!-- FIXME FIXME 23/04/01 move to experimental. Have not tested -->
The `xlearn` package has dependency on `cmake`. If one uses the `xlearn` related notebooks or scripts, make sure `cmake` is installed in the system. The easiest way to install on Linux is with apt-get: `sudo apt-get install -y build-essential cmake`. Detailed instructions for installing `cmake` from source can be found [here](https://cmake.org/install/). 

## Windows-Specific Instructions

For Spark features to work, make sure Java and Spark are installed and respective environment varialbes such as `JAVA_HOME`, `SPARK_HOME` and `HADOOP_HOME` are set properly. Also make sure environment variables `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` are set to the the same python executable.

## MacOS-Specific Instructions

We recommend using [Homebrew](https://brew.sh/) to install the dependencies on macOS, including conda (please remember to add conda's path to `$PATH`). One may also need to install lightgbm using Homebrew before pip install the package.

If zsh is used, one will need to use `pip install 'recommenders[<extras>]'` to install \<extras\>.

For Spark features to work, make sure Java and Spark are installed first. Also make sure environment variables `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` are set to the the same python executable.
<!-- TO DO: Pytorch m1 mac GPU suppoort -->

## Setup for Developers

If you want to contribute to Recommenders, please first read the [Contributing Guide](./CONTRIBUTING.md). You will notice that our development branch is `staging`.

To start developing, you need to install the latest `staging` branch in local, the `dev` package, and any other package you want. For example, for starting developing with GPU models, you can use the following command:

```bash
git checkout staging
pip install -e .[dev,gpu]
```

You can decide which packages you want to install, if you want to install all of them, you can use the following command:

```bash
git checkout staging
pip install -e .[all]
```

We also provide a [devcontainer.json](./.devcontainer/devcontainer.json)
and [Dockerfile](./tools/docker/Dockerfile) for developers to
facilitate the development on
[Dev Containers with VS Code](https://code.visualstudio.com/docs/devcontainers/containers)
and [GitHub Codespaces](https://github.com/features/codespaces).

<details>
<summary><strong><em>VS Code Dev Containers</em></strong></summary>

The typical scenario using Docker containers for development is as
follows.  Say, we want to develop applications for a specific
environment, so
1. we create a contaienr with the dependencies required, 
1. and mount the folder containing the code to the container,
1. then code parsing, debugging and testing are all performed against
   the container.
This workflow seperates the development environment from your local
environment, so that your local environment won't be affected.  The
container used here for this end is called Dev Container in the
VS Code Dev Containers extension.  And the extension eases this
development workflow with Docker containers automatically without
pain.

To use VS Code Dev Containers, your local machine must have the
following applicatioins installed:
* [Docker](https://docs.docker.com/get-started/get-docker/)
* [VS Code Remote Development Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)

Then
* When you open your local Recommenders folder in VS Code, it will
  detect [devcontainer.json](./.devcontainer/devcontainer.json), and
  prompt you to **Reopen in Container**.  If you'd like to reopen,
  it will create a container with the required environment described
  in `devcontainer.json`, install a VS Code server in the container,
  and mount the folder into the container.
  + If you don't see the prompt, you can use the command
    **Dev Containers: Reopen in Container**
* If you don't have a local clone of Recommenders, you can also use
  the command **Dev Containers: Clone Repository in Container Volume**,
  and type in a branch/PR URL of Recommenders you'd like to develop
  on, such as https://github.com/recommenders-team/recommenders,
  https://github.com/recommenders-team/recommenders/tree/staging, or
  https://github.com/recommenders-team/recommenders/pull/2098.  VS
  Code will create a container with the environment described in
  `devcontainer.json`, and clone the specified branch of Recommenders
  into the container.

Once everything is set up, VS Code will act as a client to the server
in the container, and all subsequent operations on VS Code will be
performed against the container.

</details>

<details>
<summary><strong><em>GitHub Codespaces</em></strong></summary>

GitHub Codespaces also uses `devcontainer.json` and Dockerfile in the
repo to create the environment on a VM for you to develop on the Web
VS Code.  To use the GitHub Codespaces on Recommenders, you can go to
[Recommenders](https://github.com/recommenders-team/recommenders)
$\to$ switch to the branch of interest $\to$ Code $\to$ Codespaces
$\to$ Create codespaces on the branch.

</details>

<details>
<summary><strong><em>devcontainer.json & Dockerfile</em></strong></summary>

[devcontainer.json](./.devcontainer/devcontainer.json) describes:
* the Dockerfile to use with configurable build arguments, such as
  `COMPUTE` and `PYTHON_VERSION`.
* settings on VS Code server, such as Python interpreter path in the
  container, Python formatter.
* extensions on VS Code server, such as black-formatter, pylint.
* how to create the Conda environment for Recommenders in 
  `postCreateCommand`

[Dockerfile](./tools/docker/Dockerfile) is used in 3 places:
* Dev containers on VS Code and GitHub Codespaces
* [Testing workflows on AzureML](./tests/README.md)
* [Jupyter notebook examples on Docker](./tools/docker/README.md)

</details>


## Test Environments

Depending on the type of recommender system and the notebook that needs to be run, there are different computational requirements.

Currently, tests are done on **Python CPU** (the base environment), **Python GPU** (corresponding to `[gpu]` extra above) and **PySpark** (corresponding to `[spark]` extra above).

Another way is to build a docker image and use the functions inside a [docker container](#setup-guide-for-docker).

## Setup for Making a Release

The process of making a new release and publishing it to [PyPI](https://pypi.org/project/recommenders/) is as follows:

First make sure that the tag that you want to add, e.g. `0.6.0`, is added in [`recommenders.py/__init__.py`](recommenders.py/__init__.py). Follow the [contribution guideline](CONTRIBUTING.md) to add the change.

1. Make sure that the code in main passes all the tests (unit and nightly tests).
1. Create a tag with the version number: e.g. `git tag -a 0.6.0 -m "Recommenders 0.6.0"`.
1. Push the tag to the remote server: `git push origin 0.6.0`.
1. When the new tag is pushed, a release pipeline is executed. This pipeline runs all the tests again (PR gate and nightly builds), generates a wheel and a tar.gz which are uploaded to a [GitHub draft release](https://github.com/microsoft/recommenders/releases). *NOTE: Make sure you add the release tag to the [federeted credendials](tests/README.md).*
1. Fill up the draft release with all the recent changes in the code.
1. Download the wheel and tar.gz locally, these files shouldn't have any bug, since they passed all the tests.
1. Install twine: `pip install twine`
1. Publish the wheel and tar.gz to PyPI: `twine upload recommenders*`

