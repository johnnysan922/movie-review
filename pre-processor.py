import os
import json
import re

def main():
    # Short Movie Review Classification
    dir = 'short-movie-review\\data\\'
    test_path = dir + 'test'
    train_path = dir + 'train'
    vector_path = 'short-movie-review\\feature_vectors'
    vocab_path = 'movie-review-HW2\\aclImdb\\imdb.vocab'
    vocab_set = get_vocab(vocab_path, False)
    preprocess(test_path, vector_path, vocab_set)
    preprocess(train_path, vector_path, vocab_set)

    # Movie Review Classification
    dir = 'movie-review-HW2\\aclImdb\\'
    test_path = dir + 'test'
    train_path = dir + 'train'
    vector_path = 'movie-review-HW2\\feature_vectors'
    vocab_path = 'movie-review-HW2\\aclImdb\\imdb.vocab'
    vocab_set = get_vocab(vocab_path, False)
    preprocess(test_path, vector_path, vocab_set)
    preprocess(train_path, vector_path, vocab_set)

    # Exp Movie Review Classification
    dir = 'movie-review-HW2\\aclImdb\\'
    test_path = dir + 'test'
    train_path = dir + 'train'
    vector_path = 'movie-review-HW2\\exp_feature_vectors'
    vocab_path = 'movie-review-HW2\\aclImdb\\imdb.vocab'
    vocab_set = get_vocab(vocab_path, True)
    preprocess(test_path, vector_path, vocab_set)
    preprocess(train_path, vector_path, vocab_set)

def get_vocab(path, exp):
    vocab_set = set()
    with open(path, encoding="utf8") as f:
        vocab_list = f.read().split()
        for word in vocab_list:
            vocab_set.add(word)
    if exp:
        words_to_remove = ['out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', 
"weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 
'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours']
        vocab_set = vocab_set - set(words_to_remove)
    return vocab_set

def get_word_count_dict(input, vocab):
    input_list = re.sub('[^\w\s-]', '', input.lower().replace('<br />', ' ')).split(' ')
    input_word_dict = {}
    for word in input_list:
        word = word.strip('\n')
        if word in vocab:
            input_word_dict[word] = input_word_dict.get(word, 0) + 1  
    return input_word_dict

def create_json(path, file, vectors):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, file), 'w', encoding="utf8") as out:
        out.write(json.dumps(vectors, indent=4))

def preprocess(path, vector_path, vocab):
    feature_vectors = []

    for dir, sub, files in os.walk(path):
        if len(sub) == 0:
            label = os.path.basename(dir)
            for name in files:
                if name.endswith('.txt'):
                    file_name = os.path.join(dir, name)
                    with open(file_name, 'r', encoding='utf8') as file:
                        feature_vectors.append({label: get_word_count_dict(file.read(), vocab)})

    feature_file = os.path.basename(path) + '_feature_vectors.json'

    create_json(vector_path, feature_file, feature_vectors)

if __name__ == "__main__":
    main()