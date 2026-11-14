"""This makes a separate paper from checked repair-study results, keeping old studies untouched."""

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path

TITLE = 'Does Diagnostic Targeting Improve Forecast Repair?'
REFERENCES = [
    ('Wen et al. (2025). Real-Time Reasoning Agents in Evolving Environments.',
     'https://arxiv.org/abs/2511.04898'),
    ('Lehrach et al. (2025). Code World Models for General Game Playing.',
     'https://arxiv.org/abs/2510.04542'),
    ('Lou et al. (2026). AutoHarness: improving LLM agents by automatically '
     'synthesizing a code harness.', 'https://arxiv.org/abs/2603.03329'),
    ('Orvalho et al. (2025). Counterexample Guided Program Repair Using Zero-Shot '
     'Learning and MaxSAT-based Fault Localization. AAAI.',
     'https://arxiv.org/abs/2502.07786'),
    ('Zhu et al. (2025). Where LLM Agents Fail and How They can Learn From Failures.',
     'https://arxiv.org/abs/2509.25370'),
]


def percentage(value):
    if value is None:
        return 'unavailable'
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError('Expected a numeric proportion')
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Proportion outside [0, 1]')
    return f'{100 * value:.1f}%'


def signed_points(value):
    if value is None:
        return 'unavailable'
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError('Expected a numeric contrast')
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('Contrast outside [-1, 1]')
    return f'{100 * value:+.1f} pp'


def load_json(path):
    def reject_constant(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def input_receipt(paths):
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in paths}


def safe_output(path):
    path = Path(path).resolve()
    if path.name == 'diagnose-then-repair.pdf':
        raise ValueError('The previous study paper must not be overwritten')
    if path.suffix.lower() != '.pdf':
        raise ValueError('Output must be a PDF')
    return path


def paragraph(text, style):
    from reportlab.platypus import Paragraph
    return Paragraph(escape(str(text)), style)


def page_footer(canvas, document):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColorRGB(0.35, 0.35, 0.35)
    canvas.drawString(document.leftMargin, 30, 'Diagnosis-guided forecast repair')
    canvas.drawRightString(576, 30, str(document.page))
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--output', type=Path,
                        default=Path('output/pdf/diagnosis-guided-repair.pdf'))
    args = parser.parse_args()
    safe_output(args.output)
    load_json(args.summary)
    raise SystemExit('Rendering is locked until the audited summary schema is integrated.')


if __name__ == '__main__':
    main()
