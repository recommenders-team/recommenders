# Contribution Guidelines

Here you will find the contribution guidelines.

<details>
<summary><strong><em>Click here to see the Table of Contents</em></strong></summary>

* [Microsoft Contributor License Agreement](#microsoft-contributor-license-agreement)
* [Recommenders Team Contribution Guidelines](#recommenders-team-contribution-guidelines)
  * [Test Driven Development](#test-driven-development)
  * [Python and Docstrings Style](#python-and-docstrings-style)
* [Code of Conduct](#code-of-conduct)
</details>

## Microsoft Contributor License Agreement

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

## Recommenders Team Contribution Guidelines

#### Test Driven Development

We use [Test Driven Development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development) in our development. All contributions to the repository should have unit tests, we use [pytest](https://docs.pytest.org/en/latest/) for Python files and [papermill](https://github.com/nteract/papermill) for notebooks. 

Apart from unit tests, we also have nightly builds with smoke and integration tests. For more information about the differences, see a [quick introduction to unit, smoke and integration tests](https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/).

You can find a guide on how to manually execute all the tests in the [TESTs.md](TESTS.md)

#### Python and Docstrings Style
We use the automatic style formatter [Black](https://github.com/ambv/black). See the installation guide for [VSCode](https://github.com/ambv/black#visual-studio-code) and [PyCharm](https://github.com/ambv/black#pycharm).

We use [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting the docstrings.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Apart from the official Code of Conduct developed by Microsoft, in the Recommenders team we adopt the following behaviors, to create a great working environment:
- Focus on improvement not blame
- Be empathetic, ask questions don't make demands
- Provide constructive feedback with supporting evidence when applicable


