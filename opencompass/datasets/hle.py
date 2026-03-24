import json
import math
import re

import numpy as np
from datasets import load_dataset

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import get_logger

from .base import BaseDataset


@LOAD_DATASET.register_module()
class HLEDataset(BaseDataset):

    @staticmethod
    def load(path: str, category: str | None = None, filter_images: bool = True):
        dataset = load_dataset(path)
        if filter_images:
            ds = dataset['test'].filter(lambda x: x['image'] == '')
        else:
            ds = dataset['test']
        if category:
            ds = ds.filter(lambda x: x['category'] == category)
        ds = ds.rename_column('question', 'problem')
        dataset['train'] = ds
        dataset['test'] = ds
        return dataset


def _hle_parse_judge_response(judgement: str) -> dict:
    """Parse the official HLE judge output format.

    Handles both plain-text and JSON responses from the judge model.

    Returns:
        dict with keys:
            correct (bool | None): Whether the answer was judged correct.
            confidence (int | None): Extracted confidence score 0-100.
            extracted_final_answer (str | None): The answer the judge
                extracted from the model response.
            reasoning (str | None): The judge's reasoning.
    """
    result = {
        'correct': None,
        'confidence': None,
        'extracted_final_answer': None,
        'reasoning': None,
    }

    # Try JSON parse first (structured output case)
    try:
        data = json.loads(judgement)
        if isinstance(data, dict):
            if 'correct' in data:
                val = str(data['correct']).lower().strip()
                result['correct'] = val == 'yes' or val == 'true'
            if 'confidence' in data:
                result['confidence'] = min(int(data['confidence']), 100)
            if 'extracted_final_answer' in data:
                result['extracted_final_answer'] = str(data['extracted_final_answer'])
            if 'reasoning' in data:
                result['reasoning'] = str(data['reasoning'])
            return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Plain-text parsing
    # Extract "correct: yes" or "correct: no"
    match = re.search(r'\bcorrect\b\s*[:=]\s*["\']?(yes|no)["\']?',
                      judgement, re.IGNORECASE)
    if match:
        result['correct'] = match.group(1).lower() == 'yes'
    else:
        # Fallback: look for standalone yes/no near end of response
        lines = judgement.strip().splitlines()
        for line in reversed(lines):
            stripped = line.strip().lower()
            if re.search(r'\byes\b', stripped):
                result['correct'] = True
                break
            if re.search(r'\bno\b', stripped):
                result['correct'] = False
                break

    # Extract "confidence: N" (0-100)
    match = re.search(r'\bconfidence\b\s*[:=]\s*(\d+)', judgement,
                      re.IGNORECASE)
    if match:
        result['confidence'] = min(int(match.group(1)), 100)

    # Extract "extracted_final_answer: ..."
    match = re.search(
        r'extracted_final_answer\s*[:=]\s*(.+?)(?:\n|$)', judgement,
        re.IGNORECASE)
    if match:
        result['extracted_final_answer'] = match.group(1).strip().strip("'\"")

    # Extract "reasoning: ..."
    match = re.search(r'reasoning\s*[:=]\s*(.+?)(?=\n\w+\s*[:=]|\Z)',
                      judgement, re.IGNORECASE | re.DOTALL)
    if match:
        result['reasoning'] = match.group(1).strip()

    return result


def _hle_calib_err(confidence, correct, p='2', beta=100):
    """Compute Expected Calibration Error (ECE).

    Ported from the official HLE implementation:
    https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_judge_results.py
    Original source:
    https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py

    Args:
        confidence: numpy array of confidence scores in [0, 1].
        correct: numpy array of boolean correctness values.
        p: norm to use ('1', '2', or 'infty').
        beta: target bin size.

    Returns:
        float: calibration error.
    """
    if len(confidence) < beta:
        # Not enough samples for even one full bin; use single bin
        diff = abs(float(np.nanmean(confidence)) -
                   float(np.nanmean(correct)))
        return diff

    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta]
            for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            if p == '2':
                cerr += (num_examples_in_bin / total_examples * np.square(difference))
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p in ('infty', 'infinity', 'max'):
                cerr = np.maximum(cerr, difference)

    if p == '2':
        cerr = np.sqrt(cerr)

    return float(cerr)


@DICT_POSTPROCESSORS.register_module('hle_judge_postprocess')
def hle_judge_postprocess(output: dict, output_path: str) -> dict:
    """Postprocessor that computes official HLE benchmark metrics.

    Computes the same metrics as the official HLE evaluation:
    https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_judge_results.py

    Metrics:
        accuracy: Percentage of correct answers over total questions.
        accuracy_ci: 95% confidence interval half-width (Wald estimator).
        calibration_error: Expected Calibration Error (ECE, L2, beta=100).
    """
    logger = get_logger()

    correct_list = []
    confidence_list = []
    details = []
    parse_failures = 0

    for k, v in output.items():
        judge_text = v.get('prediction', '')
        parsed = _hle_parse_judge_response(judge_text)

        is_correct = parsed['correct'] if parsed['correct'] is not None else False
        # Default confidence to 100 if not available, matching official implementation
        conf = parsed['confidence'] if parsed['confidence'] is not None else 100

        if parsed['correct'] is None:
            parse_failures += 1

        correct_list.append(is_correct)
        confidence_list.append(conf)

        details.append({
            'pred': judge_text,
            'gold': v.get('gold', ''),
            'correct': is_correct,
            'confidence': conf,
            'extracted_final_answer': parsed['extracted_final_answer'],
            'reasoning': parsed['reasoning'],
        })

    if parse_failures > 0:
        logger.warning(
            f'Failed to parse judge correctness for {parse_failures}/'
            f'{len(output)} samples (treated as incorrect)')

    n = len(output)
    correct_arr = np.array(correct_list, dtype=float)
    confidence_arr = np.array(confidence_list, dtype=float) / 100.0

    # Accuracy (same as official: correct / total * 100)
    accuracy = round(100 * float(np.sum(correct_arr)) / n, 2) if n > 0 else 0.0

    # 95% confidence interval half-width (Wald estimator)
    accuracy_ci = round(
        1.96 * math.sqrt(accuracy * (100 - accuracy) / n),
        2) if n > 0 else 0.0

    # Calibration error (ECE, L2, beta=100)
    calibration_error = round(
        100 * _hle_calib_err(confidence_arr, correct_arr, p='2', beta=100),
        2) if n > 0 else 0.0

    correct_count = int(np.sum(correct_arr))

    return {
        'accuracy': accuracy,
        'accuracy_ci': accuracy_ci,
        'calibration_error': calibration_error,
        'total_correct': correct_count,
        'total_count': n,
        'parse_failures': parse_failures,
        'details': details,
    }
