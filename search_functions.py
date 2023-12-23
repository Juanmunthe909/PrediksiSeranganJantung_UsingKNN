# search_functions.py
import difflib
from urllib.parse import urljoin

def string_to_binary(input_string):
    return ' '.join(format(ord(char), '08b') for char in input_string)

def contains_word(title, word):
    return word.lower() in title.lower()

def binary_search_title_recursive(df, target_title_binary, start, end):
    if start > end:
        return False

    mid = (start + end) // 2
    title_binary = df.loc[mid, 'title_binary']

    if title_binary == target_title_binary:
        return True
    elif title_binary < target_title_binary:
        return binary_search_title_recursive(df, target_title_binary, mid + 1, end)
    else:
        return binary_search_title_recursive(df, target_title_binary, start, mid - 1)

def binary_search_title(df, target_title_binary, start, end):
    return binary_search_title_recursive(df, target_title_binary, start, end)

def search_in_dataframe(df, search_term):
    search_term_binary = string_to_binary(search_term)
    matching_rows = df[df['title'].apply(lambda x: contains_word(x, search_term))]
    return matching_rows

def find_similar_binary(df, target_title_binary):
    for index, row in df.iterrows():
        similarity_ratio = difflib.SequenceMatcher(None, row['title_binary'], target_title_binary).ratio()
        if similarity_ratio > 0.8:  # Ubah threshold sesuai kebutuhan
            return df.loc[index]
    return None
