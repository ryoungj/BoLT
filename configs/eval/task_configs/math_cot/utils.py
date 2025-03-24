import re
import signal
from typing import Dict, List, Optional

import datasets

from lm_eval.utils import eval_logger
from math_verify import parse, verify
import numpy as np


try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )



LATENT_PREFIX = "<|START_OF_LATENT|><|PRIOR_PREFIX|>"
LATENT_SUFFIX = "<|END_OF_LATENT|>"



# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"


def doc_to_text_for_latent(doc: dict) -> str:  
    # # remove the solution prefix which is included in the latent
    # return doc_to_text(doc) + LATENT_PREFIX
    return "Problem:" + "\n" + doc["problem"] + "\n" + LATENT_PREFIX


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            ),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
        },
    ]


def list_fewshot_samples_synthetic() -> list[dict]:
    return  [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "To find the domain of the expression, we need to ensure the numerator and denominator are both defined and the denominator is not zero. \n\nStart with the numerator: $\\sqrt{x-2}$. This square root is defined when the expression inside is non-negative. Thus, we require:\nx - 2 \u2265 0\nx \u2265 2.\n\nNext, consider the denominator: $\\sqrt{5-x}$. This square root is defined when the expression inside is non-negative as well, and it must also be strictly positive to avoid division by zero. Therefore, we need:\n5 - x > 0\nx < 5.\n\nNow, we combine the two inequalities. From the first inequality, we have x \u2265 2, and from the second inequality, we have x < 5. \n\nThe combined conditions are:\n2 \u2264 x < 5.\n\nIn interval notation, this is expressed as [2, 5). \n\nThus, the final answer is [2, 5).\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "To find the determinant of the product of two matrices, we use the property that states $\\det(\\mathbf{A} \\mathbf{B}) = \\det(\\mathbf{A}) \\cdot \\det(\\mathbf{B})$. This property holds for any square matrices $\\mathbf{A}$ and $\\mathbf{B}$ of the same size. \n\nGiven that $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12$, we can directly apply this property. \n\nWe calculate:\n\n1. Start with the known determinants: $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12$.\n2. Multiply these two values together: $2 \\cdot 12$.\n3. Perform the multiplication: $2 \\cdot 12 = 24$.\n\nThus, based on these calculations, we conclude that $\\det(\\mathbf{A} \\mathbf{B}) = 24$.\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "To determine how many times Terrell must lift the two 15-pound weights to equal the total weight lifted with the two 20-pound weights, we first calculate the total weight lifted with the 20-pound weights. \n\nEach 20-pound weight contributes 20 pounds, and since he lifts two weights, the total weight per lift is 20 pounds + 20 pounds = 40 pounds. He lifts this total 12 times, resulting in a total weight of 40 pounds * 12 = 480 pounds. \n\nNext, we compute how many times he needs to lift the two 15-pound weights to match this total weight. Each 15-pound weight contributes 15 pounds, so the total weight per lift with the 15-pound weights is 15 pounds + 15 pounds = 30 pounds. \n\nTo find the number of lifts required to reach the same total weight of 480 pounds, we set up the equation: 30 pounds * x lifts = 480 pounds, where x represents the number of lifts. \n\nSolving for x, we divide both sides by 30 pounds: \nx = 480 pounds / 30 pounds = 16 lifts. \n\nThus, Terrell must lift the two 15-pound weights 16 times to equal the total weight he lifts with the two 20-pound weights.\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "To find the ratio $\\frac{a}{b}$, we start with the system of equations given. The first equation is $6x - 4y = a$, and the second is $6y - 9x = b$. \n\nWe can express $a$ in terms of $x$ and $y$ from the first equation:\na = 6x - 4y.\n\nNext, we rearrange the second equation to express $b$:\nb = 6y - 9x.\n\nTo find the ratio $\\frac{a}{b}$, we substitute the expressions we derived:\n$\\frac{a}{b} = \\frac{6x - 4y}{6y - 9x}$.\n\nNext, we need to simplify this expression. We can factor out a common factor in the numerator and the denominator. First, observe that both $a$ and $b$ can be rewritten in a way that may reveal their relationship:\nIn the numerator, we can rearrange it as $6x - 4y = 2(3x - 2y)$.\nIn the denominator, we rearrange $6y - 9x = 3(2y - 3x)$.\n\nNow, substituting these factorizations back into our ratio gives us:\n$\\frac{a}{b} = \\frac{2(3x - 2y)}{3(2y - 3x)}$.\n\nNext, we can simplify further. Notice that $2y - 3x$ can be rewritten as $-(3x - 2y)$:\n$\\frac{a}{b} = \\frac{2(3x - 2y)}{3(-(3x - 2y))}$.\n\nThis simplifies to:\n$\\frac{a}{b} = \\frac{2}{-3} = -\\frac{2}{3}$.\n\nSince we assumed $b$ is nonzero, this leads us to conclude that the system of equations has a specified relationship between $a$ and $b$ when both $x$ and $y$ are nonzero.\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
        },
    ]


def convert_to_latent_format(few_shot_examples: list[dict]) -> list[dict]:
    for example in few_shot_examples:
        # Two changes:
        # 1. Insert the latent suffix
        # 2. Include the final answer sentence in the latent thought (so that the model can learn to output the correct final answer as in the latent thought)
        final_answer = re.search(r"Final Answer: (.*). I hope it is correct.", example["solution"]).group(1)
        example["solution"] = example["solution"].replace("\nFinal Answer:", f"\n{final_answer}\n{LATENT_SUFFIX} Final Answer:")
    
    return few_shot_examples


def list_fewshot_samples_for_latent() -> list[dict]:
    few_shot_examples = list_fewshot_samples()
    few_shot_examples = convert_to_latent_format(few_shot_examples)
    return few_shot_examples


def list_fewshot_samples_for_latent_synthetic() -> list[dict]:
    few_shot_examples = list_fewshot_samples_synthetic()
    few_shot_examples = convert_to_latent_format(few_shot_examples)
    return few_shot_examples



# def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
#     candidates = results[0]

#     unnormalized_answer = get_unnormalized_answer(candidates)
#     answer = normalize_final_answer(unnormalized_answer)

#     if is_equiv(answer, doc["answer"]):
#         retval = 1
#     else:
#         retval = 0

#     results = {
#         "exact_match": retval,
#     }
#     return results


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    gold = parse(doc["solution"])

    results = results[0]
    if isinstance(results, str):
        answer = parse(results)
        correct = verify(gold, answer)

        results = {
            "exact_match": correct,
        }
    else:
        all_answers = [parse(result) for result in results]
        num_samples = len(results)
        
        log_spaced_nums = [int(2 ** exponent) for exponent in range(int(np.floor(np.log2(num_samples))))]
        if num_samples not in log_spaced_nums:
            log_spaced_nums.append(num_samples)

        # Pre-compute all correctness checks against gold answer
        all_corrects = np.array([verify(gold, answer) for answer in all_answers])

        # Pre-compute pairwise comparison matrix
        comparison_matrix = np.zeros((num_samples, num_samples), dtype=bool)
        for i in range(num_samples):
            for j in range(i + 1, num_samples):  # Only compute upper triangle
                comparison_matrix[i, j] = verify(all_answers[i], all_answers[j])
                comparison_matrix[j, i] = comparison_matrix[i, j]  # Matrix is symmetric
            comparison_matrix[i, i] = True  # An answer agrees with itself

        results = {}
        for k in log_spaced_nums:
            # For each k, only consider the first k samples
            k_matrix = comparison_matrix[:k, :k]
            agreement_scores = k_matrix.sum(axis=1)
            
            # Majority@k: Check if the most agreed-upon answer is correct
            majority_idx = np.argmax(agreement_scores)
            majority_correct = all_corrects[majority_idx]  # Use pre-computed result
            results[f"maj@{k}"] = int(majority_correct)
            
            # Pass@k: Check if any of the first k answers are correct
            any_correct = np.any(all_corrects[:k])  # Use pre-computed results
            results[f"pass@{k}"] = int(any_correct)

        # Add the original exact_match metric (equivalent to majority@1)
        results["exact_match"] = results["pass@1"]

        
    return results


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer
