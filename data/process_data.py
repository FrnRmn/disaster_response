import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath:str, categories_filepath:str) -> pd.DataFrame:
    '''                                                         #!!! Should not be Dataframe, yet already convert them into values
        Load data from two file paths, merge them into a single data frame.

        Args
        messages_filepath (str): file path of messages .csv file
        categories_filepath (str): file path of categories .csv file

        Returns
        Dataframe of the merged datasets
    
    '''
    # Read data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge data
    return messages.merge(categories, how='left', on='id')


def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    '''
        Clean data frame categories columns

        Args
        df (DataFrame): Dataframe with messy 'categories' column

        Returns
        df (DataFrame): Dataframe with one column for each category in previous 'categories' column 
        and without duplicates

    '''
    # Create dataframe with one column for each category in the 'categories' column
    categories = df['categories'].str.split(';', expand=True)

    # Rename each column with the category name (removing non-text parts)
    categories.rename(columns=categories.loc[0,:].replace('-(\w+)', '',regex=True), inplace=True)

    # Replace each entry with the boolean value relative to each category (removing text parts)
    categories.replace(r'\D', '',regex=True, inplace=True)

    # Combining previous dataframe with new categories dataframe
    df = pd.concat([df.drop(columns=['categories']), categories], axis=1)

    # Remove duplicate rows
    return df.drop_duplicates()


def save_data(df: pd.DataFrame, database_filename:str) -> None:
    '''
        Save given dataframe data into given database

        Args
        df (DataFrame): pandas DataFrame containings data to save
        database_filename (str): file path of the database file
    
    '''
    #create engine using sqlalchemy
    engine = create_engine('sqlite:///'+database_filename)

    #save data to database
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()