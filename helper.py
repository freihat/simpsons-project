import pandas as pd
import pickle
def load_data(data_dir):
    data = pd.read_csv(data_dir, error_bad_lines=False, index_col=0)
    #fillling na values
    data.character_id.fillna(0, inplace=True)
    data.location_id.fillna(0, inplace=True)
    data.raw_character_text.fillna('', inplace=True)
    data.raw_location_text.fillna('', inplace=True)
    data.spoken_words.fillna('', inplace=True)
    data.normalized_text.fillna('', inplace=True)
    data.word_count.fillna(0, inplace=True)
    #casting proprt data types
    data = data[data.timestamp_in_ms != 'Springfield Elementary School']
    data.timestamp_in_ms=data.timestamp_in_ms.astype(int)
    data.speaking_line=data.speaking_line.astype(bool)
    data.character_id=data.character_id.astype(int)
    data.location_id=data.location_id.astype(int)
    data.word_count=data.word_count.astype(int)
    # sort according to index
    data.sort_index(inplace=True)
    return data

def preprocess_and_save_data(text, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))

def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


