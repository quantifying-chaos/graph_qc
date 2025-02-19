# Managing jupyter notebook with jupytext.

Jupyter notebook intergrated source code files with binary outputs, making each notebook file enormous and unsuitable for committing into git. 
Jupytext solves this problem by converting notebook files into short concise python files. 

To install jupytext

```bash 
pip install jupytext 
uv tool install jupytext
```

Here is a short cheatsheet to use jupytext

```bash
# Convert notebook to python file 
jupytext --to py notebook.ipynb 
# Convert python file to notebook 
jupytext --to notebook python.py
```

If using `uv`, prefix the command with `uvx`

```bash
uvx jupytext --to py notebook.ipynb
uvx jupytext --to notebook python.py
```

Check more info on [jupytext](https://jupytext.readthedocs.io/en/latest/using-cli.html) and [uv](https://docs.astral.sh/uv/).
