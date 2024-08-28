#!/bin/bash

which python

conda activate py38

# Data | Info comparison
python exp/info_comparison/tree_model.py
python exp/info_comparison/linear_model.py

# Data | FE comparison
python exp/FE_comparison/dnn.py
python exp/FE_comparison/xgb.py

# Data | Frequency comparison
# python exp/freq_comparison/launch.py

# Model | Architecture comparison
python exp/model_comparison/tree_model.py
python exp/model_comparison/temporal_model.py
python exp/model_comparison/st_model.py

# Model | Objective comparison
python exp/objective_comparison/dtml.py
python exp/objective_comparison/lstm.py
python exp/objective_comparison/rgcn.py

# Eval | Robustness
# python exp/robustness/launch.py

# Eval | Decay
# python exp/decay/launch.py

# Eval | Overfitting
# python exp/overfit/launch.py

# Eval | Hyperparameter tuning
# python exp/hptune/launch.py
