import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, make_scorer, recall_score, accuracy_score, precision_score


def load_data(database_filepath: str) -> tuple[np.ndarray, np.ndarray, list]:
    '''
        Load data from a database file paths, divide features and targets and return their values.

        Args
        database_filepath (str): file path of database

        Returns
        X (ndarray) = features values
        Y (ndarray) = targets values
        category_names (list) = list of targets category names
    '''

    # Create sqlalchemy engine for the given database
    engine = create_engine('sqlite:///'+database_filepath)

    # Load database table in dataframe
    df = pd.read_sql_table('messages', engine)

    # Divide features and targets
    X = df.loc[:,'message']
    Y = df.iloc[:,4:]

    # Remove 'child_alone' column becasue it contains only zeros
    Y = Y.drop(columns=['child_alone'])

    # Extract targets category names
    category_names = Y.columns.tolist()

    # In 'related' column replace 2 with 0 because in both cases all the other column values are zero
    Y[Y['related'] == '2'] = '0'

    # Extract values from dataframes
    X = X.values
    Y = Y.astype(int).values

    return X, Y, category_names


def tokenize(text:str)->list[str]:
    '''
        Transform a string text into a list of relevant word tokens

        Args
        text (str): String data to clean and convert into tokens

        Returns
        tokens (list): A list of tokens representative of the input text
    '''

    # Normalization
    text = re.sub(r"[^a-z0-9]", " ", text.lower())

    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop-words
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    
    # Lemmatization
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    
    # Stemming
    tokens = [PorterStemmer().stem(token) for token in tokens]  
    
    return tokens


def build_model():
    '''
        Define the model pipeline and the cross-validation gridsearch

        Returns
        model (class): gridsearch object ready to be fitted
    '''

    # Define the pipeline
    pipeline = Pipeline([
        ('count_vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf_trans', TfidfTransformer()),
        ('classify', MultiOutputClassifier(RandomForestClassifier(class_weight="balanced_subsample")))
    ])

    # Define different parameters for the grid search
    parameters = {
    'classify__estimator__n_estimators' : [100, 150],
    'classify__estimator__min_samples_leaf': [2, 5],
    }

    # Define different scores for the grid search
    scorers = {
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'accuracy': make_scorer(accuracy_score),
    }

    # Define the grid search
    model = GridSearchCV(pipeline, param_grid=parameters, scoring=scorers, refit='recall', verbose=2, cv=3)

    return model


def evaluate_model(model:Pipeline , X_test:np.ndarray, Y_test:np.ndarray, category_names:list):
    '''
        Predict test targets given test features and evaluate model performance printing a report

        Args
        model (Pipeline): fitted pipeline model
        X_test (ndarray): test feature values
        Y_test (ndarray): test target values
        category_names (list): list of targets category names
    '''

    # Use the model to make predictions on test data
    Y_pred = model.predict(X_test)

    # Print the model performance report
    for col in range(Y_pred.shape[1]):
        print(f"Performance on category n° {col} - {category_names[col].upper()}")
        rep = classification_report(Y_test[:,col], Y_pred[:,col])
        print(rep)


def save_model(model, model_filepath):
    '''
        Save the trained model as a pickle file

        Args
        model (Pipeline): fitted pipeline model
        model_filepath (str): filepath where to save the model
    '''

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()