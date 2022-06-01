from sqlalchemy import create_engine
import pandas as pd
import plotly
import plotly.graph_objs as go
from models.train_classifier import load_data

# engine = create_engine('sqlite:///./data/DisasterResponse.db')
# df = pd.read_sql_table('messages', engine)

# # Divide features and targets
# X = df.loc[:,'message']
# Y = df.iloc[:,4:]

# # Remove 'child_alone' column becasue it contains only zeros
# Y = Y.drop(columns=['child_alone'])

# # Extract targets category names
# category_names = Y.columns.tolist()

# # In 'related' column replace 2 with 0 because in both cases all the other column values are zero
# Y[Y['related'] == '2'] = '0'


X, Y, category_names = load_data('./data/DisasterResponse.db')
X = pd.DataFrame(X)
Y = pd.DataFrame(Y, columns=category_names)

# Count proportion of 1s for each category
one_counts = Y[Y == 1].count()
sorted_counts = one_counts.sort_values(ascending=False)

# create visuals
graph = go.Figure({
        'data': [
            go.Bar(
                y=sorted_counts,
                x=sorted_counts.index
            )
        ],

        'layout': {
            'title': 'Distribution of Categories',
            'yaxis': {
                'title': "Count of 1s"
            },
            'xaxis': {
                'title': "Categories"
            }
        }
    })

graph.show()