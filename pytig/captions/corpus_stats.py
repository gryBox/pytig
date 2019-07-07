"""
Calculates Stats on a corpus.  A describe corpus similar to pandas.dataframe.describe
TODO: Add max and other stats.  Should this be extended to include those docs?
"""
from pprint import pformat
import textacy
import pandas as pd
# doc extentions are a concept introduced in textacy 0.7.0
textacy.spacier.doc_extensions.set_doc_extensions()

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class CorpusStats:
    def __init__(self, corpus):
        """
        A minimal set of statistical descriptions needed to maximize_captions.
        TODO:  Add other stats e.g. mean, std
        """

        # self.min_sents = self.minSents(corpus)
        # self.max_sents = self.maxSents(corpus)

        # self.min_tokens = self.minTokens(corpus)
        # self.max_tokens = self.maxTokens(corpus)

        self.docstats_df = self.calc_docstats_df(corpus)
        logger.debug(f"Finished calculating CorpusStats")



    def minSents(self, corpus):
        min_sents = min([doc._.n_sents for doc in corpus])
        return min_sents

    def minTokens(self, corpus):
        min_tokens = min([doc._.n_tokens for doc in corpus])
        return min_tokens


    def maxSents(self, corpus):
        min_sents = max([doc._.n_sents for doc in corpus])
        return min_sents

    def maxTokens(self, corpus):
        min_tokens = max([doc._.n_tokens for doc in corpus])
        return min_tokens

    def calc_doc_stats(self, doc):

        doc_df = pd.DataFrame()

        # Initializes dictionary with doc metadata
        doc_record = doc._.meta

        # Add basic counts i.e m_words
        ts =  textacy.text_stats.TextStats(doc)
        doc_record.update(ts.basic_counts)

        doc_df = doc_df.append(doc_record, ignore_index=True)
        return doc_df

    def calc_docstats_df(self, crps):

        # Get a list of records with doc stats for df
        docstat_lst = [self.calc_doc_stats(doc) for doc in crps]

        # Load into df
        docstats_df = pd.concat(docstat_lst, axis=0, ignore_index=True)


        return docstats_df


    def __repr__(self):

        return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4, width=1)





# # Example Usage
# class MyClass(Printable):
#     pass

# my_obj = MyClass()
# my_obj.msg = "Hello"
# my_obj.number = "46"
# print(my_obj)