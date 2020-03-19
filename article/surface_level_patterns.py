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
results = defaultdict(list)

for template, α, dα in tqdm(generator):
    for pattern in patterns:
        means = calculate_activation_means(α, template, pattern)
        results[pattern].append(means)

# Save computation results.
absa.utils.save(results, 'results.bin')
