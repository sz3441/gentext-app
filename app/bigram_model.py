import random
from collections import defaultdict


class BigramModel:
    def __init__(self, corpus):
        self.bigram_dict = defaultdict(list)

        for sentence in corpus:
            words = sentence.split()

            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i + 1]

                self.bigram_dict[current_word].append(next_word)

    def generate_text(self, start_word, length):
        words = [start_word]
        current_word = start_word

        for _ in range(length - 1):
            possible_next_words = self.bigram_dict.get(current_word)

            if not possible_next_words:
                break

            next_word = random.choice(possible_next_words)

            words.append(next_word)
            current_word = next_word

        return " ".join(words)