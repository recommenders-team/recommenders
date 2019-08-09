# Tests

This project uses unit, smoke and integration tests with Python files and notebooks:

 * In the unit tests we just make sure the notebook runs. 
 * In the smoke tests, we run them with a small dataset or a small number of epochs to make sure that, apart from running, they provide reasonable metrics. 
 * In the integration tests we use a bigger dataset for more epochs and we test that the metrics are what we expect. 

For more information, see a [quick introduction to unit, smoke and integration tests](https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/). To manually execute the unit tests in the different environments, first **make sure you are in the correct environment as described in the [SETUP.md](../SETUP.md)**. 

The existing unit, smoke and integration tests are also run as-is on AzureML.  A separate set of pipelines was created to run the tests on AzureML. Those pipelines are used as a control plane to submit the tests to run on AzureML. The AzureML configuration information is set in each pipeline yml file and passed to a python script, submit_azureml_pytest.py, which uses the AzureML SDK to set up the environment.  A second script, run_pytest.py, is submitted to AzureML to execute pytest on AzureML. The second script uses pytest to run tests on utilities or runs papermill to execute tests on notebooks.  The same tests and testmarkers are used as described below.
Note: Spark tests are not currently run on AzureML and may be set up in the future.

## Test execution

**Click on the following menus** to see more details on how to execute the unit, smoke and integration tests:

<details>
<summary><strong><em>Unit tests</em></strong></summary>

Unit tests ensure that each class or function behaves as it should. Every time a developer makes a pull request to staging or master branch, a battery of unit tests is executed. 

**Note that the next instructions execute the tests from the root folder.**

For executing the Python unit tests for the utilities:

    pytest tests/unit -m "not notebooks and not spark and not gpu"

For executing the Python unit tests for the notebooks:

    pytest tests/unit -m "notebooks and not spark and not gpu"

For executing the Python GPU unit tests for the utilities:

    pytest tests/unit -m "not notebooks and not spark and gpu"

For executing the Python GPU unit tests for the notebooks:

    pytest tests/unit -m "notebooks and not spark and gpu"

For executing the PySpark unit tests for the utilities:

    pytest tests/unit -m "not notebooks and spark and not gpu"

For executing the PySpark unit tests for the notebooks:

    pytest tests/unit -m "notebooks and spark and not gpu"

</details>


<details>
<summary><strong><em>Smoke tests</em></strong></summary>

Smoke tests make sure that the system works and are executed just before the integration tests every night.

**Note that the next instructions execute the tests from the root folder.**

For executing the Python smoke tests:

    pytest --durations=0 tests/smoke -m "smoke and not spark and not gpu"

For executing the Python GPU smoke tests:

    pytest --durations=0 tests/smoke -m "smoke and not spark and gpu"

For executing the PySpark smoke tests:

    pytest --durations=0 tests/smoke -m "smoke and spark and not gpu"

*NOTE: Adding `--durations=0` shows the computation time of all tests.* 

</details>

<details>
<summary><strong><em>Integration tests</em></strong></summary>

Integration tests make sure that the program results are acceptable.

**Note that the next instructions execute the tests from the root folder.**

For executing the Python integration tests:

    pytest --durations=0 tests/integration -m "integration and not spark and not gpu"

For executing the Python GPU integration tests:

    pytest --durations=0 tests/integration -m "integration and not spark and gpu"

For executing the PySpark integration tests:

    pytest --durations=0 tests/integration -m "integration and spark and not gpu"

*NOTE: Adding `--durations=0` shows the computation time of all tests.* 

</details>

<details>
<summary><strong><em>Current Skipped Tests</em></strong></summary>

Several of the tests are skipped for various reasons which are noted below.

<table><tr>
<td>Test Module</td>
<td>Test</td>
<td>Test Environment</td>
<td>Reason</td>
</tr><tr>
<td>unit/test_nni</td>
<td>*</td>
<td>Windows</td>
<td>NNI is not currently supported on Windows</td>
</tr><tr>
<td>integration/test_notebooks_python</td>
<td>test_nni_tuning_svd</td>
<td>Windows</td>
<td>NNI is not currently supported on Windows</td>
</tr><tr>
<td>*/test_notebook_pyspark</td>
<td>test_mmlspark_lightgbm_criteo_runs</td>
<td>Windows</td>
<td>MML Spark and LightGBM issue: https://github.com/Azure/mmlspark/issues/483</td>
</tr><tr>
<td>unit/test_gpu_utils</td>
<td>test_get_cuda_version</td>
<td>Windows</td>
<td>Current method for retrieval of CUDA info on Windows is install specific</td>
</tr><tr>
<td>vowpalwabbit</td>
<td>Any test with vowpalwabbit</td>
<td>AzureML</td>
<td>At the time the test pipelines on AzureML were developed, a pip installable version of vowpalwabbit was not available and required C++.  C++ was not readily available in the AzureML environment so we removed these tests and will wait until an easily installed version of vowpalwabbit is available.  </td>
</tr></table>

In order to skip a test because there is an OS or upstream issue which cannot be resolved you can use pytest [annotations](https://docs.pytest.org/en/latest/skipping.html).
 
Example:

    @pytest.mark.skip(reason="<INSERT VALID REASON>")
    @pytest.mark.skipif(sys.platform == 'win32', reason="Not implemented on Windows")
    def test_to_skip():
        assert False


</details>


## How to create tests on notebooks with Papermill

In the notebooks of this repo, we use [Papermill](https://github.com/nteract/papermill) in unit, smoke and integration tests. Papermill is a tool that enables you to parameterize notebooks, execute and collect metrics across the notebooks, and summarize collections of notebooks.

### Developing unit tests with Papermill

Executing a notebook with Papermill is easy, this is what we mostly do in the unit tests. Next we show just one of the tests that we have in [tests/unit/test_notebooks_python.py](unit/test_notebooks_python.py). 

```
import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME

@pytest.mark.notebooks
def test_sar_single_node_runs(notebooks):
    notebook_path = notebooks["sar_single_node"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
```

Notice that the input of the function is a fixture defined in [conftest.py](conftest.py). For more information, please see the [definition of fixtures in PyTest](https://docs.pytest.org/en/latest/fixture.html).

For executing this test, first make sure you are in the correct environment as described in the [SETUP.md](../SETUP.md): 

**Note that the next instruction executes the tests from the root folder.**

```
pytest tests/unit/test_notebooks_python.py::test_sar_single_node_runs
```


### Developing smoke and integration tests with Papermill

A more advanced option is used in the smoke and integration tests, where we not only execute the notebook, but inject parameters and recover the computed metrics.

The first step is to tag the parameters that we are going to inject. For it we need to modify the notebook. We will add a tag with the name `parameters`. To add a tag, go the the notebook menu, View, Cell Toolbar and Tags. A tag field will appear on every cell. The variables in the cell tagged with `parameters` can be injected. The typical variables that we inject are `MOVIELENS_DATA_SIZE`, `EPOCHS` and other configuration variables for our algorithms. 

The way papermill works to inject parameters is very simple, it generates a copy of the notebook (in our code we call it `OUTPUT_NOTEBOOK`), and creates a new cell with the injected variables. 

The second modification that we need to do to the notebook is to record the metrics we want to test using `pm.record("output_variable", python_variable_name)`. We normally use the last cell of the notebook to record all the metrics. These are the metrics that we are going to control to in the smoke and integration tests.

This is an example on how we do a smoke test. The complete code can be found in [tests/smoke/test_notebooks_python.py](smoke/test_notebooks_python.py):

```
import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME

TOL = 0.05

@pytest.mark.smoke
def test_sar_single_node_smoke(notebooks):
    notebook_path = notebooks["sar_single_node"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k"),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]
    assert results["precision"] == pytest.approx(0.326617179, TOL)
    assert results["recall"] == pytest.approx(0.175956743, TOL)
```

As it can be seen in the code, we are injecting the dataset size and the top k and we are recovering the precision and recall at k. 

For executing this test, first make sure you are in the correct environment as described in the [SETUP.md](../SETUP.md): 

**Note that the next instructions execute the tests from the root folder.**

```
pytest tests/smoke/test_notebooks_python.py::test_sar_single_node_smoke
```

More details on how to integrate Papermill with notebooks can be found in their [repo](https://github.com/nteract/papermill).

