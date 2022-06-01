import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
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

# load model
model = joblib.load("../models/classifierCVrecall.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # GRAPH1 Count proportion of 1s for each category
    one_counts = Y[Y == '1'].count()
    sorted_counts = one_counts.sort_values(ascending=False)

    # GRAPH2 Distribution of messages genre
    Y_corr_mat = Y.astype(int).corr(method='pearson')

    # GRAPH3 Distribution of messages genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
                #GRAPH 1
                {
            'data': [
                Bar(
                    y=sorted_counts,
                    x=sorted_counts.index.str.replace('_', ' ')
                )
            ],

            'layout': {
                'title': 'Category Prevalence',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
                #GRAPH 2
                {
                'data': [
                    Heatmap(
                        z=Y_corr_mat,
                        type = 'heatmap',
                        x=category_names,
                        y=category_names,
                        colorscale = 'RdBu',   #'balance_r'
                        zmin=-1, 
                        zmax=1
                    )
                ],

                'layout': {
                    'heigth': 500,
                    'width': 500,
                    'title': 'Category Correlation Matrix',
                    'yaxis': {
                        'title': "Cat"
                    },
                    'xaxis': {
                        'title': "Cat"
                    }
                }
            },
                #GRAPH 3
                {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(category_names, classification_labels))


    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=list(classification_results.items())
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()