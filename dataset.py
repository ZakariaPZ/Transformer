import os 


def encode(_string, encoding_dict):
    return [encoding_dict[char] for char in _string]


def decode(char_list, decoding_dict):
    return [decoding_dict[char] for char in char_list]


def build_data(train_path, val_path=None):
    with open(train_path, 'r', encoding='utf-8') as file:
        dataset = file.read()

        chars = sorted(list(set(dataset)))
        vocab_size = len(chars)

        char_to_idx = {char: i for i, char in enumerate(chars)}
        idx_to_char = {i: char for i, char in enumerate(chars)}

        train_dataset = encode(dataset, char_to_idx)
    
    val_dataset = None
    if val_path:
        with open(val_path, 'r', encoding='utf-8') as file:
            dataset = file.read()
            val_dataset = encode(dataset, char_to_idx)
    
    return char_to_idx, idx_to_char, vocab_size, train_dataset, val_dataset
        
