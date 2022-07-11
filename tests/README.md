# Tests

In this document we show our test infrastructure and how to contribute test to the repository.

## Types of tests

This project uses unit, smoke and integration tests with Python files and notebooks:

* In the unit tests we just make sure the utilities and notebooks run.

* In the smoke tests, we run them with a small dataset or a small number of epochs to make sure that, apart from running, they provide reasonable machine learning metrics. These can be run sequentially with integration tests to detect quickly simple errors, and should be fast.

* In the integration tests we use a bigger dataset with more epochs, and we test that the machine learning metrics are what we expect. These tests can take longer.

These types of tests are integrated in the repo in two ways, via the PR gate, and the nightly builds. 

The PR gate are the set of tests executed after doing a pull request and they should be quick. Here we include the unit tests, that just check that the code doesn't have any errors.

The nightly builds tests are executed asynchronously and can take longer. Here we include the smoke and integration tests, and their objective is to not only make sure that there are not errors, but also to make sure that the machine learning solutions are doing what we expect.

For more information, see a [quick introduction to unit, smoke and integration tests](https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/).

## Test infrastructure using AzureML

AzureML is used to run the existing unit, smoke and integration tests. AzureML benefits include being able to run the tests in parallel, managing the compute environment by automatically turning it on/off, automatic logging of artifacts from test runs and more. GitHub is used as a control plane to configure and run the tests on AzureML.  

In the following figure we show a workflow on how the tests are executed via AzureML:

<img src="https://recodatasets.z20.web.core.windows.net/images/AzureML_tests.svg?sanitize=true">

GitHub workflows `azureml-unit-tests.yml`, `azureml-cpu-nightly.yml`, `azureml-gpu-nightly.yml` and `azureml-spark-nightly` located in [.github/workflows/](../.github/workflows/) are used to run the tests on AzureML. The parameters to configure AzureML are defined in the workflow yml files. The tests are divided into groups and each workflow triggers these test groups in parallel, which significantly reduces end-to-end execution time. 

There are three scripts used with each workflow, all of them are located in [test/ci/azureml_tests](./ci/azureml_tests/):

* `submit_groupwise_azureml_pytest.py`: this script uses parameters in the workflow yml to set up the AzureML environment for testing using the AzureML SDK.
* `run_groupwise_pytest.py`: this script uses pytest to run the tests of the libraries and notebooks. This script runs in an AzureML workspace with the environment created by the script above.
* `test_groups.py`: this script defines groups of tests. If the tests are part of the unit tests, the total compute time of each group should be less than 15min. If the tests are part of the nightly builds, the total time of each group should be less than 35min.

## How to create tests

In this section we show how to create tests and add them to the test pipeline. The steps you need to follow are:

1. Create your code in the library and/or notebooks.
1. Design the unit tests for the code.
1. If you have written a notebook, design the notebook tests and check that the metrics that it returns is what you expect.
1. Add the tests to the AzureML pipeline in the corresponding [test group](./ci/azureml_tests/test_groups.py). **Please note that if you don't add your tests to the pipeline, they will not be executed.**

### How to create tests for the library code

You want to make sure that all your code works before you submit it to the repository. Here are guidelines for creating the unit tests:

* It is better to create multiple small tests than one large test that checks all the code.
* Use `@pytest.fixture` to create data in your tests.
* Use the mark `@pytest.mark.gpu` if you want the test to be executed in a GPU environment. Use `@pytest.mark.spark` if you want the test to be executed in a Spark environment.
* Use `@pytest.mark.smoke` and `@pytest.mark.integration` to mark the tests as smoke tests and integration tests.
* Use `@pytest.mark.notebooks` if you are testing a notebook.
* Avoid using `is` in the asserts, instead use the operator `==`.
* Follow the pattern `assert computation == value`, for example:
```python
assert results["precision"] == pytest.approx(0.330753)
```
* Check always the limits of your computations, for example, you want to check that the RMSE between two equal vectors is 0:
```python
assert rmse(rating_true, rating_true) == 0
assert rmse(rating_true, rating_pred) == pytest.approx(7.254309)
```

### How to create tests on notebooks with Papermill and Scrapbook

In the notebooks of this repo, we use [Papermill](https://github.com/nteract/papermill) and [scrapbook](https://nteract-scrapbook.readthedocs.io/en/latest/) in unit, smoke and integration tests. Papermill is a tool that enables you to parameterize and execute notebooks. `scrapbook` is a library for recording a notebook’s data values and generated visual content as “scraps”. These recorded scraps can be read at a future time. We use `scrapbook` to collect the metrics in the notebooks.

#### Developing unit tests with Papermill and Scrapbook

Executing a notebook with Papermill is easy, this is what we mostly do in the unit tests. Next, we show just one of the tests that we have in [tests/unit/examples/test_notebooks_python.py](unit/examples/test_notebooks_python.py).

```python
import pytest
import papermill as pm

@pytest.mark.notebooks
def test_sar_single_node_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["sar_single_node"]
    pm.execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)
```

Notice that the input of the function is a fixture defined in [conftest.py](conftest.py). For more information, please see the [definition of fixtures in PyTest](https://docs.pytest.org/en/latest/fixture.html).

For executing this test, first make sure you are in the correct environment as described in the [SETUP.md](../SETUP.md): 

*Notice that the next instruction executes the tests from the root folder.*

```bash
pytest tests/unit/test_notebooks_python.py::test_sar_single_node_runs
```

#### Developing smoke and integration tests with Papermill and scrapbook

A more advanced option is used in the smoke and integration tests, where we not only execute the notebook, but inject parameters and recover the computed metrics.

The first step is to tag the parameters that we are going to inject. For it we need to modify the notebook. We will add a tag with the name `parameters`. To add a tag, go the the notebook menu, View, Cell Toolbar and Tags. A tag field will appear on every cell. The variables in the cell tagged with `parameters` can be injected. The typical variables that we inject are `MOVIELENS_DATA_SIZE`, `EPOCHS` and other configuration variables for our algorithms.

The way papermill works to inject parameters is very simple, it generates a copy of the notebook (in our code we call it `OUTPUT_NOTEBOOK`), and creates a new cell with the injected variables.

The second modification that we need to do to the notebook is to record the metrics we want to test using `sb.glue("output_variable", python_variable_name)`. We normally use the last cell of the notebook to record all the metrics. These are the metrics that we are going to control to in the smoke and integration tests.

This is an example on how we do a smoke test. The complete code can be found in [tests/smoke/examples/test_notebooks_python.py](tests/smoke/examples/test_notebooks_python.py):

```python
import pytest
import papermill as pm
import scrapbook as sb

TOL = 0.05
ABS_TOL = 0.05

@pytest.mark.smoke
def test_sar_single_node_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["sar_single_node"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k"),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]
    assert results["precision"] == pytest.approx(0.330753, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.176385, rel=TOL, abs=ABS_TOL)
```

As it can be seen in the code, we are injecting the dataset size and the top k and we are recovering the precision and recall at k. 

For executing this test, first make sure you are in the correct environment as described in the [SETUP.md](../SETUP.md): 

*Notice that the next instructions execute the tests from the root folder.*

```
pytest tests/smoke/test_notebooks_python.py::test_sar_single_node_smoke
```

More details on how to integrate Papermill with notebooks can be found in their [repo](https://github.com/nteract/papermill). Also, you can check the [Scrapbook repo](https://github.com/nteract/scrapbook).

### How to add tests to the AzureML pipeline

To add a new test to the AzureML pipeline, add the test path to an appropriate test group listed in [test_groups.py](https://github.com/microsoft/recommenders/blob/main/tests/ci/azureml_tests/test_groups.py). 

Tests in `group_cpu_xxx` groups are executed on a CPU-only AzureML compute cluster node. Tests in `group_gpu_xxx` groups are executed on a GPU-enabled AzureML compute cluster node with GPU related dependencies added to the AzureML run environment. Tests in `group_pyspark_xxx` groups are executed on a CPU-only AzureML compute cluster node, with the PySpark related dependencies added to the AzureML run environment. 

It's important to keep in mind while adding a new test that the runtime of the test group should not exceed the specified threshold in [test_groups.py](tests/ci/azureml_tests/test_groups.py).

Example of adding a new test:

1. In the environment that you are running your code, first see if there is a group whose total runtime is less than the threshold
```python
"group_spark_001": [  # Total group time: 271.13s
    "tests/smoke/recommenders/dataset/test_movielens.py::test_load_spark_df",  # 4.33s
    "tests/integration/recommenders/datasets/test_movielens.py::test_load_spark_df",  # 25.58s + 101.99s + 139.23s
],
```
2. Add the test to the group, add the time it takes to compute, and update the total group time.
```python
"group_spark_001": [  # Total group time: 571.13s
    "tests/smoke/recommenders/dataset/test_movielens.py::test_load_spark_df",  # 4.33s
    "tests/integration/recommenders/datasets/test_movielens.py::test_load_spark_df",  # 25.58s + 101.99s + 139.23s
    #
    "tests/path/to/test_new.py::test_new_function", # 300s
],
```
3. If all the groups of your environment are above the threshold, add a new group.

## How to execute tests in your local environment

To manually execute the tests in the CPU, GPU or Spark environments, first **make sure you are in the correct environment as described in the [SETUP.md](../SETUP.md)**.

*Click on the following menus* to see more details on how to execute the unit, smoke and integration tests:

<details>
<summary><strong><em>Unit tests</em></strong></summary>

Unit tests ensure that each class or function behaves as it should. Every time a developer makes a pull request to staging or main branch, a battery of unit tests is executed.

*Note that the next instructions execute the tests from the root folder.*

For executing the Python unit tests for the utilities:

    pytest tests/unit -m "not notebooks and not spark and not gpu" --durations 0

For executing the Python unit tests for the notebooks:

    pytest tests/unit -m "notebooks and not spark and not gpu" --durations 0

For executing the Python GPU unit tests for the utilities:

    pytest tests/unit -m "not notebooks and not spark and gpu" --durations 0

For executing the Python GPU unit tests for the notebooks:

    pytest tests/unit -m "notebooks and not spark and gpu" --durations 0

For executing the PySpark unit tests for the utilities:

    pytest tests/unit -m "not notebooks and spark and not gpu" --durations 0

For executing the PySpark unit tests for the notebooks:

    pytest tests/unit -m "notebooks and spark and not gpu" --durations 0

</details>

<details>
<summary><strong><em>Smoke tests</em></strong></summary>

Smoke tests make sure that the system works and are executed just before the integration tests every night.

*Note that the next instructions execute the tests from the root folder.*

For executing the Python smoke tests:

    pytest tests/smoke -m "smoke and not spark and not gpu" --durations 0

For executing the Python GPU smoke tests:

    pytest tests/smoke -m "smoke and not spark and gpu" --durations 0

For executing the PySpark smoke tests:

    pytest tests/smoke -m "smoke and spark and not gpu" --durations 0

*NOTE: Adding `--durations 0` shows the computation time of all tests.*

*NOTE: Adding `--disable-warnings` will disable the warning messages.*

</details>

<details>
<summary><strong><em>Integration tests</em></strong></summary>

Integration tests make sure that the program results are acceptable.

*Note that the next instructions execute the tests from the root folder.*

For executing the Python integration tests:

    pytest tests/integration -m "integration and not spark and not gpu" --durations 0

For executing the Python GPU integration tests:

    pytest tests/integration -m "integration and not spark and gpu" --durations 0

For executing the PySpark integration tests:

    pytest tests/integration -m "integration and spark and not gpu" --durations 0

*NOTE: Adding `--durations 0` shows the computation time of all tests.*

</details>

<details>
<summary><strong><em>Current Skipped Tests</em></strong></summary>

Several of the tests are skipped for various reasons which are noted below.

<table>
<tr>
<td>Test Module</td>
<td>Test</td>
<td>Test Environment</td>
<td>Reason</td>
</tr>
<tr>
<td>unit/recommenders/datasets/test_wikidata</td>
<td>*</td>
<td>Linux</td>
<td>Wikidata API is unstable</td>
</tr>
<tr>
<td>integration/recommenders/datasets/test_notebooks_python</td>
<td>test_wikidata</td>
<td>Linux</td>
<td>Wikidata API is unstable</td>
</tr>
<tr>
<td>*/test_notebooks_python</td>
<td>test_vw*</td>
<td>Linux</td>
<td>VW pip package has installation incompatibilities</td>
</tr>
<tr>
<td>*/test_notebooks_python</td>
<td>test_nni*</td>
<td>Linux</td>
<td>NNI pip package has installation incompatibilities</td>
</tr>
</table>

In order to skip a test because there is an OS or upstream issue which cannot be resolved you can use pytest [annotations](https://docs.pytest.org/en/latest/skipping.html).

Example:

    @pytest.mark.skip(reason="<INSERT VALID REASON>")
    @pytest.mark.skipif(sys.platform == 'win32', reason="Not implemented on Windows")
    def test_to_skip():
        assert False

</details>