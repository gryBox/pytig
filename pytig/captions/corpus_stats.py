"""
Calculates Stats on a corpus.  A describe corpus similar to pandas.dataframe.describe
TODO: Add max and other stats.  Should this be extended to include those docs?
"""
from pprint import pformat
import textacy
import pandas as pd
import dask.dataframe as dd
from dask.delayed import delayed
import dask
# doc extentions are a concept introduced in textacy 0.7.0
textacy.spacier.doc_extensions.set_doc_extensions()

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class CorpusStats:
    def __init__(self, corpus=None):
        """
        A minimal set of statistical descriptions needed to maximize_captions.
        TODO:  Add other stats e.g. mean, std
        """

        # self.min_sents = self.minSents(corpus)
        # self.max_sents = self.maxSents(corpus)

        # self.min_tokens = self.minTokens(corpus)
        # self.max_tokens = self.maxTokens(corpus)

        if corpus is not None:
            self.docstats_df = self.calc_docstats_df(corpus).compute()
            #self.describe = self.docstats_df.describe().round()
            logger.debug(f"Finished calculating CorpusStats")
        #self.docstats_dd.compute()







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

    @dask.delayed
    def make_record_df(self, doc):

        # Initializes dictionary with doc metadata
        doc_record = doc._.meta

        # Add basic counts i.e m_words
        ts =  textacy.text_stats.TextStats(doc)
        doc_record.update(ts.basic_counts)

        return pd.DataFrame([doc_record])

    def calc_docstats_df(self, crps):

        # Get a list of records with doc stats for df
        docstat_lst = [self.make_record_df(doc) for doc in crps]

        # Load into dask df
        docstats_dd = dd.from_delayed(
            docstat_lst,
            # meta=dtype_sample_df
            )


        return docstats_dd


    def __repr__(self):

        return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4, width=1)





# # Example Usage
# class MyClass(Printable):
#     pass

# my_obj = MyClass()
# my_obj.msg = "Hello"
# my_obj.number = "46"
# print(my_obj)