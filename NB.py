import json
import math
import sys

def main():
    print('Please select one of the following (enter a number 1 thru 3)')
    print('1. Small Movie Genre Classification')
    print('2. Movie Review Classification with Bag-Of-Words (BOW) Params')
    print('3. Movie Review Classification with Less Features (no stop words)')

    choice = int(input('> '))
    while type(choice) != int or choice < 1 or choice > 3:
        print('Enter a number between 1 and 3 (inclusive)')
        choice = int(input('> '))
    train, test, param, output = get_files(choice)
    train_naive_bayes(train, test, param, output, choice)
    print("Values have been populated in the output and .NB file")

def get_files(choice):
    if choice == 1:
        train = 'short-movie-review\\feature_vectors\\train_feature_vectors.json'
        test = 'short-movie-review\\feature_vectors\\test_feature_vectors.json'
        param = 'short-movie-review\\movie_review_small.NB'
        output = 'short-movie-review\\output.txt'
        return train, test, param, output
    elif choice == 2:
        train = 'movie-review-HW2\\feature_vectors\\train_feature_vectors.json'
        test = 'movie-review-HW2\\feature_vectors\\test_feature_vectors.json'
        param = 'movie-review-HW2\\movie_review_BOW.NB'
        output = 'movie-review-HW2\\output.txt'
        return train, test, param, output
    else:
        train = 'movie-review-HW2\\exp_feature_vectors\\train_feature_vectors.json'
        test = 'movie-review-HW2\\exp_feature_vectors\\test_feature_vectors.json'
        param = 'movie-review-HW2\\exp_movie_review_BOW.NB'
        output = 'movie-review-HW2\\exp_output.txt'
        return train, test, param, output


def train_naive_bayes(train, test, params, output, choice):
    f = open(train)
    train_data = json.load(f)
    f.close()

    labels = {}
    BOW = {}
    total_words = {}
    vocab_dict = {}
    model_params = {}

    for i in train_data:
        for label, review in i.items():
            if label not in labels:
                BOW[label] = {}               
            labels[label] = labels.get(label, 0) + 1
            
            for word, word_count in review.items():
                total_words[label] = total_words.get(label, 0) + word_count
                vocab_dict[word] = vocab_dict.get(label, 0) + word_count
                BOW[label][word] = BOW[label].get(word, 0) + word_count

    doc_len = len(train_data)
    for label in labels:
        key = "P(" + label + ")"
        val = labels[label]/doc_len
        model_params[key] = val
    
    total_vocab = len(vocab_dict)
    for label, word_dict in BOW.items():
        for word, word_count in word_dict.items():
            key = "P(" +  word + "|" + label + ")"
            val = (BOW[label][word] + 1)/(total_words[label] + total_vocab)
            model_params[key] = val
    
    with open(params, 'w') as outfile:
        outfile.write(json.dumps(model_params, indent=4))
    test_naive_bayes(params, test, output, labels, total_vocab, total_words, choice)    
                
def test_naive_bayes(model_params, test, output_pred, labels, total_vocab, total_words, choice):
    test_file = open(test)
    test_data = json.load(test_file)
    test_file.close()

    model_param_file = open(model_params)
    params = json.load(model_param_file)
    model_param_file.close()
    
    output = {}
    total_correct = total_incorrect = 0
    review_count = 1

    for review_vector in test_data:
        max_val = -sys.maxsize+1
        pred_label = ""

        vectors = {}
        vectors["Review"] = review_count

        for key, value in review_vector.items():
            for label in labels:
                label_name = label + " Prediction"
                
                for word, word_count in value.items():
                    prob_label = "P(" +  word   + "|" + label + ")"
                    if prob_label not in params:
                        params[prob_label] = 1 / (total_words[label] + total_vocab)
                    if label_name not in vectors:
                        vectors[label_name] = math.log2((params["P("+ label + ")"]))
                    vectors[label_name] += math.log2(params[prob_label]) * word_count

                if vectors[label_name] > max_val:
                    max_val = vectors[label_name]
                    pred_label = label
            
            vectors["Predicted"] = pred_label
            vectors["Actual"] = key

            if vectors["Predicted"] == vectors["Actual"]:
                total_correct += 1

        output[review_count] = vectors
        review_count += 1

    total_incorrect = len(test_data) - total_correct
    accuracy = (total_correct / len(test_data)) * 100
    with open(output_pred, 'w') as outfile:
        for key, val in output.items():
            if choice != 1:
                outfile.write('Review: ' + str(val['Review']).ljust(10))
                outfile.write('Negative Prediction: ' + str(round(val['neg Prediction'], 5)).ljust(15))
                outfile.write('Positive Prediction: ' + str(round(val['pos Prediction'], 5)).ljust(15))
                outfile.write('Prediction: ' + str(val['Predicted']).ljust(10))
                outfile.write('Actual: ' + str(val['Actual']).ljust(10) + '\n')
            else:
                outfile.write('Review: ' + str(val['Review']).ljust(10))
                outfile.write('Action Prediction: ' + str(round(val['action Prediction'], 5)).ljust(15))
                outfile.write('Comedy Prediction: ' + str(round(val['comedy Prediction'], 5)).ljust(15))
                outfile.write('Prediction: ' + str(val['Predicted']).ljust(10))
                outfile.write('Actual: ' + str(val['Actual']).ljust(10) + '\n')
        outfile.write('='*125 + '\n')
        outfile.write('Total Correct: ' + str(total_correct) + '\n')
        outfile.write('Total Incorrect: ' + str(total_incorrect) + '\n')
        outfile.write('Accuracy: ' + str(accuracy))

        print('Total Correct: ' + str(total_correct))
        print('Total Incorrect: ' + str(total_incorrect))
        print('Accuracy: ' + str(accuracy))

if __name__ == "__main__":
    main()
