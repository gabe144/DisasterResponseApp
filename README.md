# Disaster Response Pipeline Project
## Introduction
This project is part of Data Science Nanodegree Program by Udacity. Here we are analyzing messages that were received during disasters. Those messages are sent to disaster response organizations and usually these are the worst time to filter and select the most important messages. Only one in thousand messages is relevant for those organizations. Different organizations take care of different disasters, for example one takes care of supplies, another of water and another one of aid related items.

## Objectivies
This project aims to develop an app based on the Natural Language Processing (NLP) to classify messages and identify needs after a disaster.

## Description
The Project is divided in the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper database structure.
2. Machine Learning Pipeline to train a model able to classify text messages into appropriate categories.
3. Web App to show model results in real time.

## Prerequisities
* Python 3.5+ (I used Python 3.8)
* Machine Learning libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process libraries: NLTK
* SQLite Database libraries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

## Project Structure
There are three components for this project:

**1. ETL Pipeline:** load CSV files and clean the data according to the following order:
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

**2. ML Pipeline:** Use the machine learning pipeline to generate a supervised model that classifies the messages:
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

**3. Web App:** this app will show the classification of a message, so a organization can input a new disaster message and see the classification divided between 36 categories.

## Files
**1. app** contains the web application.

**2. data** contains the ETL pipeline (process_data.py) and the CSV input files plus the ETL pipeline output, an SQLite database.

**3. models** contains the ML pipeline (train_classifier.py) with its output, i.e. a Python pickle file with the best model from testing different classifiers and parameters. That pickle file is used in the app to classify new messages. and a Python pickle file with input for some of the graphs in the app.
