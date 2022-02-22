This is just a stripped down version of https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition for me to mess around with

Gitignore helpful rules:
 - if you are downloading new data files/setting them up h5 files
 will be ignored if they are named ___Assets.h5 within the data 
 folder. So for create datasets files name them like that so they
 don't get pushed

# Running Instructions:

## Model Folder

### On Google Colab
 - Open the Model/GoogleColabModel.ipynb jupyter notebook up with google colab, either through uploading it from your computer or conecting via GitHub
 - Set the the Runtime environment as you see fit
 - On the left hand toolbar, click on the folder icon(should be at the bottom). This will allow you to upload files
 - Upload the data/IndexFundsData.csv file, and the Model/trading_env.py file to the main folder
 - Run all the code modules
 - when prompted allow access to your google drive, as that is where it will save the results
 - right after the google drive prompt are all the model perameters, adjust them how you see fit.
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime


### On Your PC
 - Open the data/CreateIndexFundDataSet.ipynb jupyter notebook with your desired jupyter IDE, we used PyCharm and Jupyter Desktop
 - Run all the modules, and make sure it doesn't crash
 - Open the Model/PC_DDQ_LearningModel.ipynb jupyter notebook
 - Right after the imports adjust the model perameters as you see fit
 - run all the code modules
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime

## SimpleModel Folder

### On Google Colab
 - Open the SimpleModel/GoogleColabModel.ipynb jupyter notebook up with google colab, either through uploading it from your computer or conecting via GitHub
 - Set the the Runtime environment as you see fit
 - On the left hand toolbar, click on the folder icon(should be at the bottom). This will allow you to upload files
 - Upload the SimpleModel/IndexFundsData.csv file, and the SimpleModel/trading_env.py file to the main folder
 - Run all the code modules
 - when prompted allow access to your google drive, as that is where it will save the results
 - right after the google drive prompt are all the model perameters, adjust them how you see fit.
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime


### On Your PC
 - Open the SimpleModel/CreateIndexFundDataSet.ipynb jupyter notebook with your desired jupyter IDE, we used PyCharm and Jupyter Desktop
 - Run all the modules, and make sure it doesn't crash
 - Open the SimpleModel/PC_DDQ_LearningModel.ipynb jupyter notebook
 - Right after the imports adjust the model perameters as you see fit
 - run all the code modules
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime

## FXModel Folder

### On Google Colab
 - Open the FXModel/GoogleColabModel.ipynb jupyter notebook up with google colab, either through uploading it from your computer or conecting via GitHub
 - Set the the Runtime environment as you see fit
 - On the left hand toolbar, click on the folder icon(should be at the bottom). This will allow you to upload files
 - Upload the data/FXData.csv file, and the FXModel/trading_env.py file to the main folder
 - Run all the code modules
 - when prompted allow access to your google drive, as that is where it will save the results
 - right after the google drive prompt are all the model perameters, adjust them how you see fit.
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime


### On Your PC
 - Open the data/CreateFXDataSet.ipynb jupyter notebook with your desired jupyter IDE, we used PyCharm and Jupyter Desktop
 - Run all the modules, and make sure it doesn't crash
 - Open the FXModel/PC_DDQ_LearningModel.ipynb jupyter notebook
 - Right after the imports adjust the model perameters as you see fit
 - run all the code modules
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime


