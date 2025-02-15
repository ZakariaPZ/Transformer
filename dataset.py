import torch
from torch.utils.data import IterableDataset, DataLoader
from collections import defaultdict
import re 
from collections import Counter

class IterableStoryDataset(IterableDataset):
    def __init__(self, file_path, max_length=512):
        self.file_path = file_path
        self.vocab = self.build_vocab(file_path)
        self.max_length = max_length

    def build_vocab(self, file_path, min_word_freq=1):
        # Build the vocabulary (word to index map)
        vocab = {}
        vocab["<pad>"] = 0  # Padding token
        vocab["<unk>"] = 1  # Unknown token
        next_idx = 2

        word_counter = Counter()

        # Open the file and process it line by line
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Remove punctuation and split into words
                line = re.sub(r'[^\w\s]', '', line)  # Remove punctuation
                for word in line.strip().split():
                    word = word.lower()  # Normalize to lowercase
                    word_counter[word] += 1  # Count frequency of each word

        # Add words to vocab with frequency filter
        for word, freq in word_counter.items():
            if freq >= min_word_freq:  # Only include words that meet the frequency threshold
                if word not in vocab:
                    vocab[word] = next_idx
                    next_idx += 1

        return vocab

    def process_story(self, story):
        # Convert words to indices
        story_indices = [self.vocab.get(word, self.vocab["<unk>"]) for word in story]
        
        # Pad or truncate the story to the desired length
        story_indices = story_indices[:self.max_length]
        story_indices += [self.vocab["<pad>"]] * (self.max_length - len(story_indices))

        return story_indices

    def __iter__(self):
        # Read and process the file lazily, one story at a time
        with open(self.file_path, 'r') as file:
            for line in file:
                story = line.strip().split()  # Split by whitespace
                story_indices = self.process_story(story)
                yield torch.tensor(story_indices)  # Yield the processed story

# Example usage:

# Set the file path and batch size
file_path = 'TinyStories.txt'
batch_size = 8

# Initialize the dataset and dataloader
dataset = IterableStoryDataset(file_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
for batch in dataloader:
    print(batch.shape)  # Each batch will have shape (batch_size, max_length)
    # Here you can pass the batch of indices to your model's embedding layer
    breakpoint()