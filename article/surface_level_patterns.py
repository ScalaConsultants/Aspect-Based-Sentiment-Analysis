import numpy as np
from itertools import product
from collections import defaultdict
from tqdm import tqdm
import aspect_based_sentiment_analysis as absa
from article.explain import (
    get_classifier_attentions,
    calculate_activation_means
)

generator = get_classifier_attentions(
    pipeline='absa/classifier-rest-0.1',
    domain='restaurant'
)
options = ['ALL', 'NON-SPECIAL', 'CLS', 'SEP', 'ASPECT']
patterns = list(product(options, options))  # Cartesian product
means = defaultdict(list)
d_means = defaultdict(list)

for template, α, dα in tqdm(generator):
    for pattern in patterns:
        sample_means = calculate_activation_means(α, template, pattern)
        means[pattern].append(sample_means)

        sample_d_means = calculate_activation_means(dα, template, pattern)
        d_means[pattern].append(sample_d_means)

# Save computation detailed and averaged results.
absa.utils.save([means, d_means], 'detailed-results.bin')
average = lambda data: {key: np.mean(np.stack(value), axis=0)
                        for key, value in data.items()}
averaged_means = average(means)
averaged_d_means = average(d_means)
absa.utils.save([averaged_means, averaged_d_means], 'averaged-results.bin')
