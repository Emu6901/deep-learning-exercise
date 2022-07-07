Deep learning exercise:
********
::

- Preprocessing data
- Loading data
- Transform data
- Split data
- Building model
- Initialize weights & biases
- Activation function
- Loss function
- Feed forward
- Backpropagation
- Update weights
- Build trainer
- Training
- Execute (train, validate, test).
- Evaluating
- Accuracy: approx 88.2 ~ 90.2%


Installation
************
::

 git clone https://github.com/Emu6901/deep-learning-exercise.git  
 cd ./minhhnh/minhhnh
 poetry install  
 poetry config virtualenvs.in-project true 
 poetry update  

Folder Tree
***********
::

 📁 ./
 ├─📁 .vscode/
 │ └─📄 settings.json
 ├─📄 .gitignore
 ├─📄 README.md
 ├─📄 test.py
 └─📁 minhhnh/
  ├─📄 poetry.lock
  ├─📄 README.rst
  ├─📄 pyproject.toml
  ├─📁 tests/
  │ ├─📄 test_minhhnh.py
  │ └─📄 __init__.py
  └─📁 minhhnh/
    ├─📄 EDA.ipynb
    ├─📁 data/
    │ ├─📄 train_record.csv
    │ └─📄 test_record.csv
    ├─📁 .ipynb_checkpoints/
    ├─📄 main.ipynb
    ├─📁 layers/
    │ ├─📄 dense.py
    │ ├─📄 neural_network.py
    │ ├─📄 relu.py
    │ └─📄 layer.py
    ├─📄 __init__.py
    ├─📄 best_network.pkl
    ├─📁 trainer/
    │ ├─📄 __init__.py
    │ └─📄 trainer.py
    └─📁 dataset/
      ├─📄 __init__.py
      └─📄 dataset.py
 
