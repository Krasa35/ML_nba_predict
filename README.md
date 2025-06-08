# ML_nba_predict
Repository for the allNBA team prediction project. Project is part of the Machine Learning course at the Poznan University of Technology.
## Project description
The project aims to predict the allNBA and allNBA Rookies team based on player statistics and performance metrics. Early correlation analysis was held and other factors led to choosing the features for the model. Entire report written in Polish can be found in the `doc` folder. [Report link](doc/README.md)
## Project structure
- `lib` folder contains the code for the project. 
- The `data` folder contains the data used in the project. 
- The `doc` folder contains the report and other documentation. Folders related to mlflow are used to track the experiments and models. 
- `archive` folder contains the archived scripts. 
- `usage_example.ipynb` is an example of how to use the prepared functions in your own project.
- 'main.py' is the main script that finds best ml model and prints the results.
- `requirements.txt` contains the required packages to run the project.
- `final.py` is the script that donwloads the best ml model from mlflow and predicts the allNBA and allNBA Rookies teams for the 2025 season.
