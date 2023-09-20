import numpy as np
import pandas as pd
import spacy
from sklearn.preprocessing import LabelEncoder

def clean_lexile(lexile):
    """
    Function to clean lexile feature
    Args:
        lexile (str or float): The lexile measure as a string or a float

    Returns:
        int or np.nan: The cleaned lexile measure as an integer, or np.nan for 'Non-Prose' or 'nan' values
    """
    if pd.isnull(lexile):
        return np.nan
    elif isinstance(lexile, str):
        if lexile == 'Non-Prose':
            return np.nan
        else:
            # Remove the 'L' at the end and convert to integer
            return int(lexile.rstrip('L'))
    else:
        # If lexile is a float (or any non-string data type), convert to int and return
        return int(lexile)

# Function to classify author type
def classify_author(author):
    # Process the text
    doc = nlp(author)
    
    # Check if any of the entities are labeled as 'PERSON'
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return 'person'
            
    # If no 'PERSON' entity is found, return 'org'
    return 'org'


def encode_author_type(df):
    """
    Function to encode author_type feature
    Args:
        df (pd.DataFrame): The DataFrame with 'author_type' column

    Returns:
        pd.DataFrame: The DataFrame with 'author_type' replaced with numerical values
    """
    le = LabelEncoder()
    df['author_type'] = le.fit_transform(df['author_type'])
    return df

def clean_grade(df):
    """
    Function to clean grade feature
    Args:
        df (pd.DataFrame): The DataFrame with 'grade' column

    Returns:
        pd.DataFrame: The DataFrame with 'grade' replaced with integer values
    """
    df['grade'] = df['grade'].astype(str).str.replace('rd Grade', '')
    df['grade'] = df['grade'].str.replace('th Grade', '')
    df['grade'] = df['grade'].apply(lambda x: int(x) if x.isdigit() else 0)
    return df



def group_and_encode_genre(df):
    """
    Function to group and encode genre feature
    Args:
        df (pd.DataFrame): The DataFrame with 'genre' column

    Returns:
        pd.DataFrame: The DataFrame with 'genre' replaced with grouped and encoded values
    """
    genre_map = {
        'Fiction': ['Poem', 'Short Story', 'Folktale', 'Fantasy', 'Science Fiction', 'Allegory', 'Fiction - General', 'Fable', 'Myth', 'Historical Fiction', 'Magical Realism', 'Drama'],
        'Non-Fiction': ['Informational Text', 'Non-Fiction - General', 'Biography', 'Essay', 'Memoir', 'Interview', 'Psychology', 'Primary Source Document', 'Autobiography'],
        'News & Opinion': ['News', 'Opinion'],
        'Historic & Legal': ['Historical Document', 'Legal Document', 'Letter'],
        'Philosophy & Religion': ['Speech', 'Religious Text', 'Satire', 'Political Theory', 'Philosophy']
    }

    # Reverse the genre_map dictionary for mapping
    reverse_genre_map = {genre: key for key, values in genre_map.items() for genre in values}

    df['genre_big_group'] = df['genre'].map(reverse_genre_map)

    # If the genre is not found in the map, assign it to 'Other'
    df['genre_big_group'] = df['genre_big_group'].fillna('Other')

    le = LabelEncoder()
    df['genre_big_group_encode'] = le.fit_transform(df['genre_big_group'])
    
    return df



if __name__ == "__main__":
    nlp = spacy.load('en_core_web_lg')

    prompt_grade = pd.read_csv(r'./data/commonlit_texts.csv')
    # df = pd.read_csv(r'/kaggle/input/commonlit-texts/commonlit_texts.csv')
    prompt_grade['author_type'] = prompt_grade['author'].apply(classify_author)
    # prompt_grade['lexile_md'] = prompt_grade['lexile'].apply(clean_lexile)
    prompt_grade = encode_author_type(prompt_grade)
    # prompt_grade = clean_grade(prompt_grade)
    prompt_grade = group_and_encode_genre(prompt_grade)

    prompt_grade.to_csv("./data/prompt_grade.csv", index=False)