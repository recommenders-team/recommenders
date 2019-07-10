Docker Support
==============
The Dockerfile in this directory will build Docker images with all the dependencies and code needed to run example notebooks or unit tests included in this repository.

Multiple environments are supported by using [multistage builds](https://docs.docker.com/develop/develop-images/multistage-build/). In order to efficiently build the Docker images in this way, [Docker BuildKit](https://docs.docker.com/develop/develop-images/build_enhancements/) is necessary.
The following examples show how to build and run the Docker image for CPU, PySpark, and GPU environments. Once the container is running you can access Jupyter notebooks at http://localhost:8888.

Building and Running with Docker
--------------------------------

<details>
<summary><strong><em>Docker CPU environment</em></strong></summary>

```
DOCKER_BUILDKIT=1 docker build -t recommenders:cpu --build-arg ENV="cpu" .
docker run -p 8888:8888 -d recommenders:cpu
```

</details>

<details>
<summary><strong><em>Docker PySpark environment</em></strong></summary>

```
DOCKER_BUILDKIT=1 docker build -t recommenders:pyspark --build-arg ENV="pyspark" .
docker run -p 8888:8888 -d recommenders:pyspark
```

</details>

<details>
<summary><strong><em>Docker GPU environment</em></strong></summary>

```
DOCKER_BUILDKIT=1 docker build -t recommenders:gpu --build-arg ENV="gpu" .
docker run --runtime=nvidia -p 8888:8888 -d recommenders:cpu
```

By default the master branch of the repository is used, in order to change that you can provide a BRANCH build argument as shown below.

```
DOCKER_BUILDKIT=1 docker build -t recommenders:cpu --build-arg ENV="cpu" --build-arg BRANCH="staging" .
```

In order to see detailed progress with BuildKit you can provide a flag during the build command: ```--progress=plain```

Running tests with docker
-------------------------

```
docker run -it recommenders:cpu pytests tests/unit -m "not spark and not gpu and not notebooks"
```