import jupytext

# Define o caminho para o script Python e o notebook Jupyter
script_path = 'main.py'
notebook_path = 'main.ipynb'

# Carregar o script Python
with open(script_path, 'r') as file:
    python_code = file.read()

# Converter o código Python em um notebook Jupyter
notebook = jupytext.reads(python_code, format='py')

# Salvar o notebook Jupyter
jupytext.write(notebook, notebook_path)
