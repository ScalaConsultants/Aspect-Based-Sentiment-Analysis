import html
from typing import List
from typing import Tuple
import numpy as np
from IPython.core.display import display as ipython_display
from IPython.core.display import HTML
from .data_types import Pattern, PredictedExample, Review


def html_escape(text):
    return html.escape(text)


def highlight(
        token: str,
        weight: float,
        rgb: Tuple[int, int, int] = (135, 206, 250),
        max_alpha: float = 0.8,
        escape: bool = True
) -> str:
    r, g, b = rgb
    color = f'rgba({r},{g},{b},{np.abs(weight) / max_alpha})'
    span = lambda c, t: f'<span style="background-color:{c};">{t}</span>'
    token = html_escape(token) if escape else token
    html_token = span(color, token)
    return html_token


def highlight_sequence(
        tokens: List[str],
        weights: List[float],
        **kwargs
) -> List[str]:
    return [highlight(token, weight, **kwargs)
            for token, weight in zip(tokens, weights)]


def highlight_pattern(pattern: Pattern, rgb=(180, 180, 180)) -> str:
    w = pattern.importance
    html_importance = highlight(f'Importance {w:.2f}', w, rgb=rgb,
                                max_alpha=0.9)
    html_patterns = highlight_sequence(pattern.tokens, pattern.weights)
    highlighted_text = [html_importance, *html_patterns]
    highlighted_text = ' '.join(highlighted_text)
    return highlighted_text


def display_html(patterns: List[Pattern]):
    texts = []
    texts.extend([highlight_pattern(pattern) + '<br>' for pattern in patterns])
    text = ' '.join(texts)
    html_text = HTML(text)
    return html_text


def display_patterns(patterns: List[Pattern]):
    html_text = display_html(patterns)
    return ipython_display(html_text)


def display(review: Review):
    return display_patterns(review.patterns)


def summary(example: PredictedExample):
    print(f'{str(example.sentiment)} for "{example.aspect}"')
    rounded_scores = np.round(example.scores, decimals=3)
    print(f'Scores (neutral/negative/positive): {rounded_scores}')
