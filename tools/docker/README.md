Docker Support
==============
The Dockerfile in this directory will build Docker images with all
the dependencies and code needed to run example notebooks or unit
tests included in this repository.  It is also used by
* [.devcontainer/devcontainer.json](../../.devcontainer/devcontainer.json)
  to build
  [VS Code Dev Contianers](https://code.visualstudio.com/docs/devcontainers/containers)
  that can facilitate the development of Recommenders
  (See [Setup Guide](../../SETUP.md)),
* and [tests/ci/azureml_tests/aml_utils.py](../../tests/ci/azureml_tests/aml_utils.py)
  to create the environment in [the testing workflows of Recommenders](../../.github/workflows/) (See [Tests](../../tests/README.md)).

Multiple environments are supported by using 
[multistage builds](https://docs.docker.com/build/building/multi-stage/).
The following examples show how to build and run the Docker image for
CPU, PySpark, and GPU environments.

Once the container is running you can access Jupyter notebooks at
http://localhost:8888.


Building and Running with Docker
--------------------------------

* **CPU environment**

  ```bash
  docker build -t recommenders:cpu .
  docker run -v ../../examples:/root/examples -p 8888:8888 -d recommenders:cpu
  ```


* **PySpark environment**

  ```bash
  docker build -t recommenders:pyspark --build-arg EXTRAS=[spark] .
  docker run -v ../../examples:/root/examples -p 8888:8888 -d recommenders:pyspark
  ```

* **GPU environment**

  ```bash
  docker build -t recommenders:gpu --build-arg COMPUTE=gpu .
  docker run --runtime=nvidia -v ../../examples:/root/examples -p 8888:8888 -d recommenders:gpu
  ```


* **GPU + PySpark environment**

  ```bash
  docker build -t recommenders:gpu-pyspark --build-arg COMPUTE=gpu --build-arg EXTRAS=[gpu,spark] .
  docker run --runtime=nvidia -v ../../examples:/root/examples -p 8888:8888 -d recommenders:gpu-pyspark
  ```


Build Arguments
---------------

There are several build arguments which can change how the image is
built. Similar to the `ENV` build argument these are specified during
the docker build command.

Build Arg|Description|
---------|-----------|
`COMPUTE`|Compute to use, options: `cpu`, `gpu` (defaults to `cpu`)|
`EXTRAS`|Extra dependencies to use, options: `dev`, `gpu`, `spark` (defaults to none ("")); For example, `[gpu,spark]`|
`GIT_REF`|Git ref of Recommenders to install, options: `main`, `staging`, etc (defaults to `main`); Empty value means editable installation of current clone|
`JDK_VERSION`|OpenJDK version to use (defaults to `21`)|
`PYTHON_VERSION`|Python version to use (defaults to `3.11`)|
`RECO_DIR`|Path to the copy of Recommenders in the container when `GIT_REF` is empty (defaults to `/root/Recommenders`)|

Examples:
* Install Python 3.10 and the Recommenders package from the staging branch.

  ```bash
  docker build -t recommenders:staging --build-arg GIT_REF=staging --build-arg PYTHON_VERSION=3.10 .
  ```

* Install the current local clone of Recommenders and its extra 'dev' dependencies.

  ```bash
  # Go to the root directory of Recommenders to copy the local clone into the Docker image
  cd ../../
  docker build -t recommenders:dev --build-arg GIT_REF= --build-arg EXTRAS=[dev] -f tools/docker/Dockerfile .
  ```

In order to see detailed progress you can provide a flag during the
build command: ```--progress=plain```


Running tests with Docker
-------------------------

* Run the tests using the `recommenders:cpu` image built above.
  NOTE: The `recommender:cpu` image only installs the Recommenders
  package under [../../recommenders/](../../recommenders/).

  ```bash
  docker run -it recommenders:cpu bash -c 'pip install pytest; \
  pip install pytest-cov; \
  pip install pytest-mock; \
  apt-get install -y git; \
  git clone https://github.com/recommenders-team/recommenders.git; \
  cd Recommenders; \
  pytest tests/unit -m "not spark and not gpu and not notebooks and not experimental"'
  ```

* Run the tests using the `recommenders:dev` image built above.
  NOTE: The `recommenders:dev` image has a full copy of your local
  Recommenders repository.

  ```bash
  docker run -it recommenders:dev bash -c 'cd Recommenders; \
  pytest tests/unit -m "not spark and not gpu and not notebooks and not experimental"'
  ```
