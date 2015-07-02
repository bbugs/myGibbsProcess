"""
Given a model, predict words on the test set
"""


class DotWordsFile(object):
    """
    """

    def __init__(self, fname):
        self.fname = fname
        self.V = 0
        self.id2word = {}
        self.word2id = {}
        self.main()

        return

    def _read_file(self):

        with open(self.fname, 'r') as f:
            self.unique_words = [word.strip() for word in f.readlines()]

    def set_vocabulary_size(self):
        self.V = len(self.unique_words)

    def set_id2word(self):
        index = 0
        for word in self.unique_words:
            self.id2word[index] = word
            self.word2id[word] = index
            index += 1
        return

    def main(self):
        self._read_file()
        self.set_vocabulary_size()
        self.set_id2word()
        return


class DotWCFile(object):

    def __init__(self, fname):
        self.fname = fname
        self.V = 0
        self.ordered_word_counts = []
        self.main()

        return


    def _read_file(self):
        with open(self.fname, 'r') as f:
            self.word_counts = [item.strip() for item in f.readlines()]

    def _order_word_counts(self):
        for item in self.word_counts:
            word, count = item.strip().split('\t')
            self.ordered_word_counts.append((int(count), word))

        self.ordered_word_counts.sort(reverse=True)
        return

    def get_top_n_words(self, n):
        return [word for count, word in self.ordered_word_counts[0:n]]

    def get_top_n_wordids(self, n, word2id):
        top_n_wordids = []
        for word in self.ordered_word_counts[0:n]:
            top_n_wordids.append(word2id[word])
        return top_n_wordids


    def main(self):
        self._read_file()
        self._order_word_counts()
        return


class DotNames():
    """
    .names files contain the filenames of the documents
    """
    def __init__(self, fname):
        self.fname = fname
        self.filenames = []
        self.D = 0
        self.main()

        return

    def _read_file(self):
        with open(self.fname, 'r') as f:
            self.filenames = [name.strip() for name in f.readlines()]
        return

    def main(self):
        self._read_file()
        self.D = len(self.filenames)
        return


class DotDocs():
    """
    .docs files contain the word ids for each document and the frequency or occurrence
    """

    def __init__(self, fname):
        self.fname = fname
        self.raw_lines = []
        self.wordid_list_all_docs = []
        self.main()

        return

    def _read_file(self):
        with open(self.fname, 'r') as f:
            self.raw_lines = [line.strip() for line in f.readlines()]
        return

    def _get_features_from_all_lines(self):

        for raw_line in self.raw_lines:
            wordid_list = DocLine(raw_line).wordid_list  # from one document
            self.wordid_list_all_docs.append(wordid_list)


    def main(self):
        self._read_file()
        self._get_features_from_all_lines()
        return


class DocLine(object):
    """
    Line from .docs file.  Example
    7 4:1 9:1 14:1 15:1 30:1 33:1 120:1
    """

    def __init__(self, doc_string):
        self.doc_string = doc_string
        self.nwords = 0
        self.wordid_list = []
        self.main()

        return

    def _parse_line(self):
        s = self.doc_string.find(' ')
        # number of words

        try:
            self.nwords = int(self.doc_string[0:s])
        except ValueError:
            return

        features = self.doc_string[s+1:].split(' ')
        # print features
        # ['4:1', '9:1', '14:1', '15:1', '30:1', '33:1', '120:1']

        feature_list = [item.split(':') for item in features]
        # [['4', '1'], ['9', '1'],...,['30', '1'], ['33', '1'], ['120', '1']]

        self.wordid_list = [int(wordid) for wordid, freq in feature_list]

    def main(self):
        self._parse_line()
        assert self.nwords == len(self.wordid_list)


def line_2_words(wordid_list, id2word):
    """(lst, dict) -> lst
    input: [4, 9, 10, 13, 14, 24, 37, 39, 76, 90]
    output: ['long', 'dress', 'evening', 'wedding', 'bridesmaid', 'a-line', 'empire', 'one shoulder', 'pink', 'silver']
    """
    word_list = []
    for word_id in wordid_list:
        word_list.append(id2word[word_id])
    return word_list


if __name__ == '__main__':
    rpath2 = '../../DATASETS/dress_attributes/txt_represention/out_all/zappos/'
    dot_docs_fname = rpath2 + 'text_features_test_zappos_0.0.docs'
    dot_docs = DotDocs(dot_docs_fname)
    dot_names_fname = rpath2 + 'file_names_test_zappos_0.0.names'
    dot_names = DotNames(dot_names_fname)
    dot_words_fname = rpath2 + 'filtered_vocabulary_zappos_0.0.words'
    dot_words = DotWordsFile(dot_words_fname)
    dot_wc_fname = rpath2 + 'vocabulary_counts_test_zappos_0.0.wc'
    dot_wc = DotWCFile(dot_wc_fname)

    # true wordids
    print dot_docs.wordid_list_all_docs
    print dot_words.id2word

    # for l in dot_docs.wordid_list_all_docs:
    #     if len(l) == 0:
    #         print l

    word_list_all_docs = []
    for word_ids in dot_docs.wordid_list_all_docs:
        words = line_2_words(word_ids, dot_words.id2word)
        word_list_all_docs.append(words)

    print word_list_all_docs













# Load the .wc from either zappos or POS


# f = open(fname, 'r')
# word_counts = f.readlines()
#
# [wc.strip().split('\t') for wc in word_counts]
# f.close()
#
#
#
# # Order in descending order from the most frequent word to the least frequent.
# print word_counts
# print type(word_counts)

# predict: generate file with the words
