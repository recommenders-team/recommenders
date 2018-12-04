# Tests

This project use unit, smoke and integration tests with Python files and notebooks. For more information, see a [quick introduction to unit, smoke and integration tests](https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/). To manually execute the unit tests in the different environments, first **make sure you are in the correct environment as described in the [setup](/SETUP.md)**. Click on the following menus to see more details:

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