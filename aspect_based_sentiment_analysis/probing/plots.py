import html
from typing import List
from typing import Tuple
import numpy as np
from IPython.core.display import HTML
from ..data_types import PredictedExample
from ..data_types import Pattern


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


def highlight_pattern(pattern: Pattern) -> str:
    rgb = (117, 255, 104) if pattern.impact >= 0 else (250, 102, 109)
    weight = pattern.impact
    html_impact = highlight(f'impact {weight:.2f}', np.abs(weight), rgb=rgb)
    html_patterns = highlight_sequence(pattern.tokens, pattern.weights)
    highlighted_text = [html_impact, *html_patterns]
    highlighted_text = ' '.join(highlighted_text)
    return highlighted_text


def explain(example: PredictedExample):
    aspect = example.aspect_representation
    texts = [f'Words connected with the "{example.aspect}" aspect: <br>']
    texts.extend(highlight_sequence(aspect.tokens, aspect.look_at))
    texts.append('<br><br>')
    texts.append('The model uses these patterns to make a prediction: <br>')
    texts.extend([highlight_pattern(pattern) + '<br>'
                  for pattern in example.patterns])
    text = ' '.join(texts)
    html_text = HTML(text)
    return html_text
