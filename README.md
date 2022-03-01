This is just a stripped down version of https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition for me to mess around with

Gitignore helpful rules:
 - if you are downloading new data files/setting them up h5 files
 will be ignored if they are named ___Assets.h5 within the data
 folder. So for create datasets files name them like that so they
 don't get pushed

# Running Instructions:


## SimpleModel Folder

### On Google Colab
 - Open the SimpleModel/GoogleColabModel.ipynb Jupyter Notebook up with Google Colab, either through uploading it from your computer or conecting via GitHub
 - Set the Runtime environment as you see fit
 - On the left hand toolbar, click on the folder icon(should be at the bottom). This will allow you to upload files
 - Upload the SimpleModel/IndexFundsData.csv file, and the SimpleModel/trading_env.py file to the main folder
 - Run all the code modules
 - when prompted allow access to your Google Drive, as that is where it will save the results
 - right after the Google Drive prompt are all the model parameters, adjust them how you see fit.
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime


### On a PC
 - Open the SimpleModel/CreateIndexFundDataSet.ipynb Jupyter Notebook with your desired Jupyter IDE, we used PyCharm and the Jupyter Notebook web application
 - Run all the modules, and make sure it doesn't crash
 - Open the SimpleModel/PC_DDQ_LearningModel.ipynb Jupyter Notebook
 - Right after the imports adjust the model parameters as you see fit
 - run all the code modules
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime

## IndexModel Folder

### On Google Colab
 - Open the IndexModel/GoogleColabModel.ipynb Jupyter notebook up with Google Colab, either through uploading it from your computer or connecting via GitHub
 - Set the Runtime environment as you see fit
 - On the left-hand toolbar, click on the folder icon(should be at the bottom). This will allow you to upload files
 - Upload the data/IndexFundsData.csv file, and the IndexModel/trading_env.py file to the main folder
 - Run all the code modules
 - when prompted allow access to your Google Drive, as that is where it will save the results
 - right after the Google Drive prompt are all the model parameters, adjust them how you see fit.
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime


### On a PC
 - Open the data/CreateIndexFundDataSet.ipynb Jupyter Notebook with your desired Jupyter IDE, we used PyCharm and the Jupyter Notebook web application
 - Run all the modules, and make sure it doesn't crash
 - Open the IndexModel/PC_DDQ_LearningModel.ipynb Jupyter Notebook
 - Right after the imports adjust the model parameters as you see fit
 - run all the code modules
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime

## IndexIntradayModel Folder

### On Google Colab
 - Open the IndexIntradayModel/GoogleColabModel.ipynb Jupyter notebook up with Google Colab, either through uploading it from your computer or connecting via GitHub
 - Set the Runtime environment as you see fit
 - On the left-hand toolbar, click on the folder icon(should be at the bottom). This will allow you to upload files
 - Upload the data/IndexFundsDataIntraday.csv file, and the IndexIntradayModel/trading_env.py file to the main folder
 - Run all the code modules
 - when prompted allow access to your Google Drive, as that is where it will save the results
 - right after the Google Drive prompt are all the model parameters, adjust them how you see fit.
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime


### On a PC
 - Open the data/CreateIndexFundIntradayDataSet.ipynb Jupyter Notebook with your desired Jupyter IDE, we used PyCharm and the Jupyter Notebook web application
 - Run all the modules, and make sure it doesn't crash
 - Open the IndexIntradayModel/PC_DDQ_LearningModel.ipynb Jupyter Notebook
 - Right after the imports adjust the model parameters as you see fit
 - run all the code modules
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime


## FXModel Folder

### On Google Colab
 - Open the FXModel/GoogleColabModel.ipynb Jupyter Notebook up with Google Colab, either through uploading it from your computer or conecting via GitHub
 - Set the Runtime environment as you see fit
 - On the left hand toolbar, click on the folder icon(should be at the bottom). This will allow you to upload files
 - Upload the data/FXData.csv file, and the FXModel/trading_env.py file to the main folder
 - Run all the code modules
 - when prompted allow access to your Google Drive, as that is where it will save the results
 - right after the Google Drive prompt are all the model parameters, adjust them how you see fit.
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime


### On a PC
 - Open the data/CreateFXDataSet.ipynb Jupyter Notebook with your desired Jupyter IDE, we used PyCharm and the Jupyter Notebook web application
 - Run all the modules, and make sure it doesn't crash
 - Open the FXModel/PC_DDQ_LearningModel.ipynb Jupyter Notebook
 - Right after the imports adjust the model parameters as you see fit
 - run all the code modules
 - Note: once you run the "Create and Initialize Environment" module if you ever need to re-run the modules, you will have to restart the runtime

