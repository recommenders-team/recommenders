# Tests

This project uses unit, smoke and integration tests with Python files and notebooks. For more information, see a [quick introduction to unit, smoke and integration tests](https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/). To manually execute the unit tests in the different environments, first **make sure you are in the correct environment as described in the [SETUP.md](/SETUP.md)**. 

## Test execution

Click on the following menus to see more details on how to execute the unit, smoke and integration tests:

<details>
<summary><strong><em>Unit tests</em></strong></summary>

Unit tests ensure that each class or function behaves as it should. Every time a developer makes a pull request to staging or master branch, a battery of unit tests is executed. 

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

For executing the Python smoke tests:

    pytest tests/smoke -m "smoke and not spark and not gpu"

For executing the Python GPU smoke tests:

    pytest tests/smoke -m "smoke and not spark and gpu"

For executing the PySpark smoke tests:

    pytest tests/smoke -m "smoke and spark and not gpu"

</details>

<details>
<summary><strong><em>Integration tests</em></strong></summary>

Integration tests make sure that the program results are acceptable

For executing the Python integration tests:

    pytest tests/integration -m "integration and not spark and not gpu"

For executing the Python GPU integration tests:

    pytest tests/integration -m "integration and not spark and gpu"

For executing the PySpark integration tests:

    pytest tests/integration -m "integration and spark and not gpu"

</details>


## How to create tests on notebooks with Papermill

In the notebooks of these repo we use [papermill](https://github.com/nteract/papermill) in unit, smoke and integration tests. 

In the unit tests we just make sure the notebook runs. In the smoke tests, we run them with a small dataset or a small number of epochs to make sure that, apart from running, they provide reasonable metrics. Finally, in the integration tests, we use a bigger dataset for more epochs and we test that the metrics are what we expect. 

Executing a notebook with papermill is easy, this is what we mostly do in the unit tests:

```
import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME

def test_my_notebook_unit_test():
    notebook_path = "path/to/my_notebook.ipynb"
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
```

For executing this code, we just need to use pytest: `pytest test_unit_my_notebook.py`.

A more advanced option is used in the smoke and integration tests, where we not only execute the notebook, but inject parameters and recover the computed metrics.

The first step is to tag the parameters that we are going to inject. For it we need to modify the notebook. We will add a tag with the name `parameters`. To add a tag, go the the notebook menu, View, Cell Toolbar and Tags. A tag field will appear on every cell. The variables in the cell tagged with `parameters` can be injected. The typical variables that we inject are `MOVIELENS_DATA_SIZE`, `EPOCHS` and other configuration variables for our algorithms. 

The way papermill works to inject the parameters is very simple, it generates a copy of the notebook (in our code we call it `OUTPUT_NOTEBOOK`), and creates a new cell with the injected variables. 

The second modification that we need to do to the notebook is to record the metrics we want to test using `pm.record("output_variable", python_variable_name)`. We usually use the last cell of the notebook to record all the metrics. These are the metrics that we are going to control to in the smoke and integration tests.

The final step is to create the smoke test:

```
import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME

TOL = 0.05

def test_my_notebook_smoke_test():
    notebook_path = "path/to/my_notebook.ipynb"
    pm.execute_notebook(notebook_path, 
                        OUTPUT_NOTEBOOK, 
                        kernel_name=KERNEL_NAME,
                        parameters=dict(MOVIELENS_DATA_SIZE="100k", EPOCHS=1))
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]
    assert results["precision"] == pytest.approx(0.326617179, TOL)
    assert results["recall"] == pytest.approx(0.175956743, TOL)
```

As it can be seen in the code, we are injecting the dataset size and the number of epochs and we are recovering the precision and recall. For executing this code, we just need to use pytest: `pytest test_smoke_my_notebook.py`.

More details on how to integrate papermill with notebooks can be found in the [repo](https://github.com/nteract/papermill).

