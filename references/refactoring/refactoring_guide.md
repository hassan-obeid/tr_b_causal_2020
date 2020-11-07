# Refactoring jupyter notebooks: End Goal
In my opinion, a Jupyter notebook should do something.

Ideally, each notebook demonstrates one or more steps of the project workflow.

In this sense, we should imagine Jupyter notebooks to be [integration tests](https://www.fullstackpython.com/integration-testing.html).
Additionally, in standard testing style, through Jupyter + Jupytext, we should also think of Jupyter notebooks as just another `.py` file.

# Final structure of the test (i.e. the notebook)
The general structure of a test is that you:
1. set up the input data, if any, for the test
2. import the function(s) to be tested
3. Execute the functions with the desired arguments and keyword arguments
4. Test that the function results are as expected.

Your notebooks, at the end, should have this format.

# Refactoring Process
Though the state described above is where you should end up, we are of course not there.
Below, I'll describe a process for going from the sort of notebooks one has after completing an initial prototype to the type of notebook described above. In other words, I'll describe my suggested refactoring process.

The description comes in the form of a table shown below.
Each row describes the state of the notebook, along three dimensions.
From row to row, only one of the dimensions changes.
The idea is that each row represents the result of one refactoring step, where the step is to change the notebook by changing the desired dimension.
For instance, the first refactoring step (from row 1 to 2) is to add programmatic tests to your notebook.
And soon and so forth.

## Evolving notebook states
|   | Data/Parameters  | Logic                                   | Testing                       |
|---|------------------|-----------------------------------------|-------------------------------|
| 1 | In notebook      | In notebook                             | Visual, in notebook           |
| 2 | In notebook      | In notebook                             | Programmatic, in notebook     |
| 3 | In notebook      | Outside notebook, in private functions  | Programmatic, in notebook     |
| 4 | Outside notebook | Outside notebook, in private functions  | Programmatic, in notebook     |
| 5 | Outside notebook | Outside notebook, in `main` function    | Programmatic, in notebook     |
| 6 | Outside notebook | Outside notebook, in `main` function    | Programmatic CLI, in notebook |

## Notes
First off, note we are at row 1 since each of you currently look at your results in the notebook and declare them good (or not).

The second step is to add some assertions that programmatically encode checks for whether the results are as expected.
See the [pytest examples](https://docs.pytest.org/en/latest/index.html) to easily get started with tests.

Three, when writing the `.py` files for the workflow, here are my suggestions:
- Try to have 1 file per one major step of the workflow.
- Encapsulate all the logic for that step in one function.
- Think carefully about what classes in one's code should do.
  Note / declare the purpose of each class with relation to established [python design patterns](https://refactoring.guru/design-patterns/python).
- Strive to write small (100 lines or less) functions.
  Make liberal use of functions inside other functions.
- Strive to write small files (1000 lines or less, hopefully less).
- Try to, in general, follow PEP8 for style.
  - Use [flake8](https://flake8.pycqa.org/en/latest/) to find violations of PEP8 in one's code.
  - Use [autopep8](https://github.com/hhatto/autopep8) to automatically fix many of these errors.
- Have a code block for `if __name__ == '__main__'` to allow one's final file that implements the workflow step to be [used as a script](https://stackoverflow.com/questions/419163/what-does-if-name-main-do) at the command line.
- Use [click](https://click.palletsprojects.com/en/7.x/) to make one's single important function (see the first suggestion) a command line utility.
 - Use [attrs](https://www.attrs.org/en/stable/overview.html) to define your classes.
 - Use [mypy type hints](http://mypy-lang.org/examples.html) to provide type annotations to your code (aka allow for static type checking to catch bugs early).

Fourth, see this general and **EXCELLENT** [guide to refactoring in general](https://refactoring.guru/refactoring).
You will learn all you need to begin and so much more to keep you going.
The material is presented in a most easy-to-understand fashion with helpful images.

Okay, that should be all for now.
More comments will come on your pull-requests as you refactor.
I hope this helps!
