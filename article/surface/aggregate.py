import numpy as np
import aspect_based_sentiment_analysis as absa

α, dα = absa.utils.load('attentions.bin')
a, da = α.sum(axis=(0, 1)), dα.sum(axis=(0, 1))

patterns = {}
layers = len(a)
for i in range(layers):

    if i == 0:
        p = np.diag(np.ones(len(a[i])))
        p += a[i]
    else:
        p = patterns[i - 1] @ a[i] + patterns[i - 1]

    patterns[i] = p
