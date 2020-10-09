import csv
import spacy
import pandas as pd
from glob import glob
import numpy as np
import os
from scispacy.abbreviation import AbbreviationDetector
from spacy import displacy
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# nlp = spacy.load("en_core_sci_md")
nlp = spacy.load("en_core_web_sm")

# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)

cxr_parsing = pd.read_csv("/home/ubuntu/simclr_/MIMIC_cxr_report.csv")


for itr in cxr_parsing.iloc[:, 1]:
    modi = itr + str(" don't can't won't")
    # print("modi",modi)
    doc = nlp(modi)
    print('Original Article: %s' % modi)
    print("\n")
    print("Word piece model : %s" % tokenizer.tokenize(modi))

    tokens = [token.text for token in doc]
    print("\n")
    input('scispacy token: %s' % tokens)

    # print("Parsed doc", doc)
    print("\n");print("\n")

    # for token in doc:
        # print(token)
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

    # displacy.serve(doc, style="dep")

    # print("Abbreviation", "\t", "Definition")
    # for abrv in doc._.abbreviations:
    #     print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")

