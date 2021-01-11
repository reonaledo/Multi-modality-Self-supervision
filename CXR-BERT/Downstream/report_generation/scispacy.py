import spacy
import pandas as pd
from glob import glob
import numpy as np
import os
from scispacy.abbreviation import AbbreviationDetector

nlp = spacy.load("en_core_sci_sm")

# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)

# Path = glob("/home/ubuntu/mimic_jpg/files/*/*/*.txt")
# def Report_to_lines(Report):
#     Report_analysis = open('MIMIC_cxr_report.csv','w')
#     wr = csv.writer(Report_analysis)
#     for i in range(len(Report)):

#         f, f_re = open(Report[i],'r'), open(Report[i],'r')
#         lines, data = f.readlines(), f_re.read() # 1) list of lines in txt
#         for idx, line in enumerate(lines):
#             if 'FINDINGS' not in data and 'IMPRESSION' not in data:
#                 wr.writerow([Report[i][-13:],'None'])
#                 print(Report[i][-13:], 'None')
#                 break

#             elif 'FINDINGS' in line:
#                 report = lines[idx:]
#                 wr.writerow([Report[i][-13:],report])
#                 print(Report[i][-13:], report)
#                 break

#             elif 'IMPRESSION' in line:
#                 report = lines[idx:]
#                 wr.writerow([Report[i][-13:],report])
#                 print(Report[i][-13:], report)
#                 break
#         f.close()
#         f_re.close()
#     Report_analysis.close()

# Report_to_lines(Path)

cxr_parsing = pd.read_csv("/home/ubuntu/simclr_/MIMIC_cxr_report.csv")

print(cxr_parsing.head)

input("Stop moment")

doc = nlp("Spinal and bulbar muscular atrophy (SBMA) is an \
           inherited motor neuron disease caused by the expansion \
           of a polyglutamine tract within the androgen receptor (AR). \
           SBMA can be caused by this easily.")

print("Abbreviation", "\t", "Definition")
for abrv in doc._.abbreviations:
	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")