import numpy as np
from scipy import stats
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
    domain='restaurant',
    limit=100
)
options = ['ALL', 'NON-SPECIAL', 'CLS', 'SEP', 'ASPECT']
patterns = list(product(options, options))  # Cartesian product
efficiency = []
means = defaultdict(list)
d_means = defaultdict(list)

for template, α, dα in tqdm(generator, total=100):

    # Measure the efficiency (normalized entropy) of attentions.
    m, n, i, j = α.shape
    entropy = stats.entropy(α.reshape(m, n, -1), axis=-1)
    efficiency.append(entropy / np.log(i * j))

    for pattern in patterns:
        sample_means = calculate_activation_means(α, template, pattern)
        means[pattern].append(sample_means)

        # Collect absolute values of gradients.
        sample_d_means = calculate_activation_means(dα, template, pattern)
        d_means[pattern].append(np.abs(sample_d_means))

# Save computation detailed and averaged results.
absa.utils.save([efficiency, means, d_means], 'detailed-results.bin')

apply = lambda func, data: {key: func(value) for key, value in data.items()}
average = lambda x: np.mean(np.stack(x), axis=0)

averaged_means = apply(average, means)
averaged_d_means = apply(average, d_means)
averaged_efficiency = average(efficiency)
absa.utils.save([averaged_efficiency, averaged_means, averaged_d_means],
                'averaged-results.bin')
