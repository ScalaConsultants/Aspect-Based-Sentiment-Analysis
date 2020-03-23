# Model Performance
# https://stanfordnlp.github.io/stanza/performance.html
# https://github.com/UniversalDependencies/UD_English-EWT
# https://universaldependencies.org
# https://universaldependencies.org/u/dep/index.html

import stanza
# # download English model
# stanza.download('en')
# # initialize English neural pipeline
# nlp = stanza.Pipeline('en')
# # run annotation over a sentence
# doc = nlp("Barack Obama was born in Hawaii. He is good.")
from stanza import Document
from stanza.utils.conll import CoNLL

fname = 'data/UD_English-EWT/en_ewt-ud-train.conllu'
with open(fname, 'r') as f:
    conll = CoNLL.load_conll(f)

sentences = CoNLL.convert_conll(conll)
document = Document(sentences)
