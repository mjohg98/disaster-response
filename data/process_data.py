# import statements
import sys 
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath): 

    """
    Loads in messages and categories, and merges into a single 
    dataframe by message id 

    Args: 
        messages_filepath: filepath of messages csv file
        categories_filepath: filepath of categories csv file 

    Returns: 
        df: merged pandas dataframe 

    """

    # import message and category dataframes
    messages = pd.read_csv(f'{messages_filepath}')
    cat = pd.read_csv(f'{categories_filepath}')

    # merge messages and categories 
    df = messages.merge(cat, how='inner', on='id')

    return df 


def clean_data(df): 

    """
    Cleans dataset
    
    Args: 
        df: original dataframe containing messages and category data 
    
    Returns: 
        df: cleaned dataframe 

    """

    categories = df['categories'].str.split(';', expand=True)

    # defining labels of categories 
    row = categories.head(1)
    category_colnames = [row[ii][0][:-2] for ii in range(row.shape[1])]

    # relablling the columns of categories dataframe
    categories.columns = category_colnames

    for column in categories: 
        # set each value to be the last character of each string 
        categories[column] = [x[-1] for x in categories[column]]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')

    # drop the original categories column 
    df.drop('categories', axis=1, inplace=True)

    # join 'df' and categories at columns
    df = pd.concat((df, categories), axis=1)

    # remove duplicate rows 
    df.drop_duplicates(inplace=True)

    return df 


def save_data(df, database_filepath): 

    """
    Creates SQLite databases and saves cleaned dataframe to database

    Args: 
        df: dataframe to save into SQLite database
        database_filepath: filepath of where to create SQLite datbase 
    
    """

    # create sqlite database
    engine = create_engine(f'sqlite:///{database_filepath}')

    # save df to database 
    df.to_sql('table', engine, if_exists='replace', index=False)


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