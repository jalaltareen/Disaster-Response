# Disaster-Response

## Project Overview 

This project is for developing an API for classification of disaster response messages using the data from figure eight. 

I created a machine learning pipeline to categorize the events so that messages can be classified and sent to respective relief agency.

It includes a web app where an emergency worker can input a new message and get classification results on several categories. Project is divided into three components:

  - ETL Pipeline as process_data.py
  - ML Pipeline as train_classifier.py  
  - Flask Web App 
 
**process_data.py** is a python script for ETL pipeline it writes a data cleaning pipeline that loads and merge the messages and categories datasets cleans it and stores it in a SQLite database.

**train_classifier.py** is a python script for writing a machine learning pipeline. It loads data from the SQLite database. Splits the dataset into training and test sets Builds a text processing and machine learning pipeline Trains and tunes a model. In end saves the final model as a pickle file

**Flask Web App** A webpage or API which takes the user input message and classify them into categories.

Instructions:
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db To run 

ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

If clicking the above link page doesn't work. Try the following steps if you are working out of **Udacity workspace**

  - Run your app with python run.py command
  - Open another terminal and type **env|grep WORK** this will give you the spaceid (it will start with view*** and some characters after that)
  - Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id that you got in the step 2
  - Press enter and the app should now run for you
