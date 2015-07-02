
from gibbs_input import process_gibbs_input as gibbs_input


# .words file
# dot_word_fname = '../../DATASETS/dress_attributes/txt_represention/out_all/zappos/filtered_vocabulary_zappos_0.0.words'
#
# dot_words = gibbs_input.DotWordsFile(dot_word_fname)

# print "vocabulary size", dot_words.V
# print "id2word", dot_words.id2word
# print "word2id", dot_words.word2id

# .wc file
# fname = '../../DATASETS/dress_attributes/txt_represention/out_all/zappos/vocabulary_counts_train_val_zappos_0.0.wc'
#
# dot_wc = gibbs_input.DotWCFile(fname)
# print dot_wc.get_top_n_words(10)
# print dot_wc.ordered_word_counts[0:10]
#
#
# # .names file
# fname = '../../DATASETS/dress_attributes/txt_represention/out_all/zappos/file_names_train_val_zappos_0.0.names'
#
# dot_names = gibbs_input.DotNames(fname)
# print dot_names.filenames[0:10]

# # .docs file
# fname = '../../DATASETS/dress_attributes/txt_represention/out_all/zappos/text_features_train_val_zappos_0.0.docs'
#
# dot_docs = gibbs_input.DotDocs(fname)
# line = dot_docs.lines[0]
#
# dot_docs.get_features_from_line(line)

# doc line
doc_string = '10 4:2 9:3 10:2 13:1 14:1 24:1 37:1 39:3 76:1 90:2'
doc_line = gibbs_input.DocLine(doc_string)
print doc_line.nwords
print doc_line.wordid_list

# doc_string = 'B00HCXSMOE 0:0'
# doc_line = gibbs_input.DocLine(doc_string)
# print doc_line.nwords
# print doc_line.wordid_list


fname = '../../DATASETS/dress_attributes/txt_represention/out_all/zappos/filtered_vocabulary_zappos_0.0.words'
gin = gibbs_input.DotWordsFile(fname=fname)
print gibbs_input.line_2_words(doc_line.wordid_list, gin.id2word)