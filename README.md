### Preview
![alt-text](https://github.com/FrnRmn/disaster_response/blob/d1f7c0dfe0988aa680d6ad6db2ee2e0103e7a041/data/disaster_example.gif)



### Dependencies
All the required packages are contained in the requirements.txt file. Use <code>pip install requirements.txt</code> to install all of them.



### Motivation
The current work is a project proposed by Udacity in collaboration with [Figure Eight Inc.](https://www.figure-eight.com/) (now Appen).

The general idea is to build a ML pipeline able to classify messages sent during disaster events into meaningful classes (e.g., Food, Medical Help, Shelter, etc...).
During emergency situations a lot of people need the support of disaster relief agencies. However, given the overwhelming number of messages it could be hard to filter the relevant messages and to automatically send the message to the suitable organization. That is why a classification pipeline could be a very useful tool.

The model used is a Random Forest Classifier and it is included in a full-stack web application built upon Flask.



### Files Description
- app<br>
| - template<br>
| |- master.html    # main page of web app<br>
| |- go.html    # classification result page of web app<br>
|- run.py   # launcher fo the web app<br>
- data<br>
|- disaster_categories.csv    # data to process<br>
|- disaster_messages.csv    # data to process<br>
|- process_data.py    # data preprocessing and database creation<br>
|- disaster_example.gif   # gif preview<br>
- models<br>
|- train_classifier.py    # model training, evaluation and saving<br>
- README.md   # the current file<br>
- requiriements.txt   #contains dependencies<br>



### Instructions
1) Open terminal in the project's root directory.
2) To preprocess the data and save them in a database file called "DisasterResponse" use <code>python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.dby</code>.
3) To train the model and save it as a piclke file called "classifier" use <code>python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl</code>
4) Use <code>cd app</code>
5) To launch the application use <code>python run.py</code>.



### Credits
All the data used for training were made available by [Figure Eight Inc.](https://www.figure-eight.com/).

The code inside the /app folder was modified starting from source code provided by [Udacity](https://www.udacity.com/).
