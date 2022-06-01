### Preview

![alt-text](https://github.com/FrnRmn/disaster_response/blob/d1f7c0dfe0988aa680d6ad6db2ee2e0103e7a041/data/disaster_example.gif)
<br>


### Dependencies
- pandas
- numpy
- plotly
- nltk
- sklearn
- flask
- sqlalchemy
- joblib
- json
- pickle


### Motivation
The current work is a project proposed by Udacity in collaboration with [Figure Eight Inc.](https://www.figure-eight.com/) (now Appen).

The general idea is to build a ML pipeline able to classify messages sent during disaster events into meaningful classes (e.g., Food, Medical Help, Shelter, etc...).
The model used is a Random Forest Classifier and it is included in a full-stack web application built upon Flask.


### Files Description and Useful Commands
- The training data are contained in the /data folder. To preprocess the data and save them in a database file called "DisasterResponse" use <code>python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.dby</code> in the project's root directory.
- The script for training the model pipeline is contained in the /models folder. To train the model and save it as a piclke file called "classifier" use <code>python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl</code>
- The webb app is contained in the /app folder. To launch the application use <code>python run.py</code>.


### Credits
All the data used for training were made available by [Figure Eight Inc.](https://www.figure-eight.com/).

The code inside the /app folder was modified starting from source code provided by [Udacity](https://www.udacity.com/).
