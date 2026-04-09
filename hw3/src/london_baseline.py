# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_corpus_path', type=str, default='birth_dev.tsv')
    args = parser.parse_args()

    predictions = ["London"] * len(open(args.eval_corpus_path, encoding='utf-8').readlines())
    total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        accuracy = correct / total * 100
        print(f'Correct: {correct} out of {total}: {accuracy}%')
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
