import os
import glob
import re
import json
import yaml
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

SIZE_DEFAULT = 14
SIZE_MEDIUM = 16
SIZE_LARGE = 18
plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels



SYNTHETIC_RAW_TOKEN_RATIO_MAP = {
    "raw": 1.0,
    "latent": 2.3 + 1.,  # +1 accounts for the added raw token
    "surface_cot": 2.3,
    "wrap_base": 0.4807564654, 
    "wrap_cot": 0.7007504097,
    "warmstart": 2.3 + 1.,  # +1 accounts for the added raw token
}


VAL_SET_ALIAS = {
    "dclm_val": "dclm_baseline_sample_final_fixed_val_split",
    "finemath_4plus_val": "finemath_4plus_final_fixed_val_split",
}

HOLDOUT_VAL_METRIC_MAP = {
    "finemath_4plus_val.nll_per_token": "Validation NLL",
    "finemath_4plus_val.elbo_4_per_token": "Validation ELBO",
    # "finemath_4plus_val.elbo_1_per_token",
    # "finemath_4plus_val.elbo_2_per_token",
    "dclm_val.nll_per_token": "Validation NLL on DCLM",
    # "dclm_val.elbo_1_per_token",
    # "dclm_val.elbo_2_per_token",
    # "dclm_val.elbo_4_per_token",
}

HOLDOUT_VAL_METRICS = list(HOLDOUT_VAL_METRIC_MAP.keys())


DOWNSTREAM_EVAL_METRIC_MAP = {
    "hendrycks_math_cot_synthetic.exact_match": "MATH (Synthetic Few-Shot CoT)",
    "hendrycks_math_cot.exact_match": "MATH (Minerva Few-Shot CoT)",
    "gsm8k_cot_alt.exact_match": "GSM8K (Default Few-Shot CoT)",
    "gsm8k_cot_synthetic_alt.exact_match": "GSM8K (Synthetic Few-Shot CoT)",
    "mmlu_cot_synthetic_stem.exact_match": "MMLU-STEM (Synthetic Few-Shot CoT)",
    "mmlu_cot_flan_stem.exact_match": "MMLU-STEM (FLAN Few-Shot CoT)",
}

DOWNSTREAM_EVAL_METRICS = list(DOWNSTREAM_EVAL_METRIC_MAP.keys())

MATH_EVAL_SUBJECTS = [
    "prealgebra",
    "algebra",
    "intermediate_algebra",
    "counting_and_prob",
    "geometry",
    "num_theory",
    "precalc",
]


DOWNSTREAM_VAL_METRICS = [
    "math.nll_per_token",
    "gsm8k.nll_per_token",
    "mmlu_stem.nll_per_token",
    "s1k_long_cot.nll_per_token",
]

VAL_METRICS = HOLDOUT_VAL_METRICS + DOWNSTREAM_VAL_METRICS

ALL_MATH_EVAL_LEVELS = [1,2,3,4,5]
PLOT_MATH_EVAL_LEVELS = [1,2,3]
DETAILED_MATH_EVAL_METRICS = []
ALL_MATH_EVAL_METRICS = []
for math_eval in ["hendrycks_math_cot_synthetic", "hendrycks_math_cot"]:
    ALL_MATH_EVAL_METRICS.append(f"{math_eval}.exact_match")
    ALL_MATH_EVAL_METRICS.append(f"latent_{math_eval}.exact_match")
    
    for level in PLOT_MATH_EVAL_LEVELS:
        DETAILED_MATH_EVAL_METRICS.append(f"{math_eval}_sub_l{level}.exact_match")
        # LATENT_METRIC_MAP[f"{math_eval}_sub_l{level}"] = f"latent_{math_eval}_sub_l{level}"
        ALL_MATH_EVAL_METRICS.append(f"{math_eval}_sub_l{level}.exact_match")
        ALL_MATH_EVAL_METRICS.append(f"latent_{math_eval}_sub_l{level}.exact_match")

    for subject in MATH_EVAL_SUBJECTS:
        DETAILED_MATH_EVAL_METRICS.append(f"{math_eval}_{subject}.exact_match")
        # LATENT_METRIC_MAP[f"{math_eval}_{subject}"] = f"latent_{math_eval}_{subject}"
        ALL_MATH_EVAL_METRICS.append(f"{math_eval}_{subject}.exact_match")
        ALL_MATH_EVAL_METRICS.append(f"latent_{math_eval}_{subject}.exact_match")


MMLU_STEM_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
]


DETAILED_MMLU_EVAL_METRICS = []
for subject in MMLU_STEM_SUBJECTS:
    DETAILED_MMLU_EVAL_METRICS.append(f"mmlu_cot_synthetic_{subject}.exact_match")
    DETAILED_MMLU_EVAL_METRICS.append(f"mmlu_cot_flan_{subject}.exact_match")

DETAILED_GSM8K_EVAL_METRICS = []
for gsm_setup in ["gsm8k_cot_alt", "gsm8k_cot_synthetic_alt"]:
    for metric in ["exact_match", "latent_exact_match"]:
        DETAILED_GSM8K_EVAL_METRICS.append(f"{gsm_setup}.{metric}")
        DETAILED_GSM8K_EVAL_METRICS.append(f"latent_{gsm_setup}.{metric}")

METRICS = HOLDOUT_VAL_METRICS + DOWNSTREAM_EVAL_METRICS

ALL_EVAL_METRICS = DOWNSTREAM_EVAL_METRICS + ALL_MATH_EVAL_METRICS + DETAILED_GSM8K_EVAL_METRICS + DETAILED_MMLU_EVAL_METRICS
ALL_EVAL_METRIC_LABEL_MAP = {**HOLDOUT_VAL_METRIC_MAP, **DOWNSTREAM_EVAL_METRIC_MAP}



BASE_EXP_DIR = "../exp_logs/"

WARMSTART_NAME = "latent_warmstart"
DEFAULT_EVAL_RESULT_FILE_NAME = "metrics.eval.jsonl"
DEFAULT_EVAL_SUBDIR = "evals"
EVAL_RESULT_FILE_NAME = "results.json"

def post_process_metrics(all_metrics, process_type="smoothed", metric_cols=METRICS, sort_by_col="step", trial_col="trial", run_col="run"):
    # Create a copy to avoid modifying the original dataframe
    processed_metrics = all_metrics.copy()
    
    for metric in metric_cols:
        # Process each run separately
        for run_name, run_data in processed_metrics.groupby(run_col):
            run_data = run_data[[sort_by_col, metric, trial_col]].dropna()
            
            # Process each trial within the run
            processed_trials = []
            for trial_num, trial_data in run_data.groupby(trial_col):
                # Sort by specified column within each trial
                trial_data = trial_data.sort_values(sort_by_col)
                
                # Apply processing based on type
                if process_type == "smoothed":
                    window_size = 5  # Adjust this value to control smoothing amount
                    trial_data[metric] = trial_data[metric].rolling(
                        window=window_size, 
                        center=True, 
                        min_periods=1
                    ).mean()
                elif process_type == "envelope":
                    # Plot the current best performance
                    if metric in VAL_METRICS and "dclm" not in metric:
                        trial_data[metric] = trial_data[metric].expanding().min()
                    else:
                        trial_data[metric] = trial_data[metric].expanding().max()
                
                processed_trials.append(trial_data)
            
            if len(processed_trials) > 0:
                # Combine processed trials back together
                processed_run = pd.concat(processed_trials)
                
                # Update only the metric column in the original dataframe
                processed_metrics.loc[processed_run.index, metric] = processed_run[metric]

    return processed_metrics


def load_run_metrics(load_runs, load_metrics=METRICS, warmstart_ckpt=None, eval_result_dirs=None, load_all_trials=True, load_latent_metrics=True):
    # gather all metrics from all runs at all steps

    all_data = []  # Use a flat list structure instead of defaultdict
    missing_metrics = defaultdict(lambda: defaultdict(list))

    for run_name, run_dir in load_runs.items():
        if load_all_trials:
            run_dir_all_trials = []

            # Base run
            if os.path.exists(os.path.join(BASE_EXP_DIR, run_dir)):
                run_dir_all_trials.append((run_dir, 0))
            
            # Additional trials
            matches = glob.glob(os.path.join(BASE_EXP_DIR, run_dir + "_trial_[0-9]*"))
            for match in matches:
                # Find all occurrences of iter_X or trial_X and take the last one
                match = match.replace(BASE_EXP_DIR, "")

                all_matches = re.findall(r'(?:trial)_(\d+)', match)
                if all_matches:
                    trial_num = int(all_matches[-1])  # Take the last match

                if os.path.exists(os.path.join(BASE_EXP_DIR, match, "config.yaml")):
                    run_dir_all_trials.append((match, trial_num))

            run_dir_all_trials.sort(key=lambda x: x[1])  # Sort by trial number
            # print(run_dir_all_trials)
            print(f"Loading {len(run_dir_all_trials)} trials for {run_name}")
        else:
            if os.path.exists(os.path.join(BASE_EXP_DIR, run_dir)):
                run_dir_all_trials = [(run_dir, 0)]
            elif os.path.exists(os.path.join(BASE_EXP_DIR, run_dir + "_trial_0")):
                run_dir_all_trials = [(run_dir + "_trial_0", 0)]
            else:
                raise ValueError(f"No trial directory found for {run_name}")

        for run_trial_dir, trial_num in run_dir_all_trials:
            run_path = os.path.join(BASE_EXP_DIR, run_trial_dir)
            load_metric_paths = []

            if eval_result_dirs is None:
                # default
                if os.path.exists(os.path.join(run_path, DEFAULT_EVAL_RESULT_FILE_NAME)):
                    load_metric_paths.append((os.path.join(run_path, DEFAULT_EVAL_RESULT_FILE_NAME), os.path.join(run_path, DEFAULT_EVAL_SUBDIR)))
            else:
                for subdir in eval_result_dirs:
                    if subdir == "":
                        eval_result_dir = os.path.join(run_path, DEFAULT_EVAL_SUBDIR)
                    else:
                        eval_result_dir = os.path.join(run_path, subdir)

                    load_metric_paths.append((os.path.join(run_path, subdir, DEFAULT_EVAL_RESULT_FILE_NAME), eval_result_dir))


            if "pretrained" not in run_name.lower():
                train_config_path = os.path.join(run_path, "config.yaml")
                with open(train_config_path, "r") as f:
                    train_config = yaml.load(f, Loader=yaml.FullLoader)
                batch_size = train_config["data"]["batch_size"] * train_config["distributed"]["dp_shard"]
                seq_len = train_config["data"]["seq_len"]
                token_per_step = batch_size * seq_len

            for (metric_path, eval_result_dir) in load_metric_paths:
                if not os.path.exists(metric_path):
                    print(f"Warning: {metric_path} does not exist")
                    continue

                with open(metric_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        step_results = {
                            "run": run_name,
                            "step": data["global_step"] if "pretrained" not in run_name.lower() else np.nan,
                            "num_tokens": data["global_step"] * token_per_step if "pretrained" not in run_name.lower() else np.nan,
                            "trial": trial_num,
                        }

                        if run_name == WARMSTART_NAME and step_results["step"] > warmstart_ckpt["step"]:
                            # skip warmstart runs after the warmstart checkpoint
                            continue

                        for metric in load_metrics:
                            if metric in ALL_EVAL_METRICS:
                                alias, metric_name = metric.split(".")
                                key = "eval/" + alias

                                if "latent" in run_name.lower() and load_latent_metrics:
                                    key = "eval/" + f"latent_{alias}"
                                    # print("Replacing metric", metric, "with", key)

                                if key not in data:
                                    # print(f"Warning: {key} not found in {run_name} (step {step_results['step']})")
                                    missing_metrics[metric][run_name].append(step_results["step"])
                                    continue
                                
                                results = data[key]
                                if "acc" in metric_name or "hendrycks_math" in alias:
                                    suffix = ",none"
                                elif "latent_exact_match" in metric_name:
                                    if "latent" not in run_name.lower():
                                        continue
                                    metric_name = "exact_match"
                                    suffix = ",latent-flexible-extract"
                                elif "exact_match" in metric_name:
                                    suffix = ",flexible-extract"
                                else:
                                    raise ValueError(f"Unknown metric: {metric}")
                                step_results[metric] = results[metric_name + suffix]
                                step_results[metric + "_stderr"] = results[metric_name + "_stderr" + suffix]
                            elif metric in VAL_METRICS:
                                val_set, metric_name = metric.split(".")
                                result_dict = data.get(f"val/{VAL_SET_ALIAS.get(val_set, val_set)}", {})
                                if metric_name not in result_dict:
                                    # print(f"Warning: {metric} not found in {run_name} (step {step_results['step']})")
                                    missing_metrics[metric][run_name].append(step_results["step"])
                                    continue
                                step_results[metric] = - result_dict[metric_name]  # negative likelihood
                            else:
                                raise ValueError(f"Unknown metric: {metric}")

                        # check if the step results already exists
                        existing_step_results = list(filter(lambda x: x["run"] == run_name and x["trial"] == trial_num and x["step"] == step_results["step"], all_data))
                        if len(existing_step_results) > 0:
                            assert len(existing_step_results) == 1, f"Multiple step results found for {run_name} (step {step_results['step']})"
                            existing_step_result = existing_step_results[0]
                            existing_step_result.update(step_results)
                        else:
                            all_data.append(step_results)

    # if missing_metrics:
    #     print("\nMissing metrics:")
    #     for metric, missing_entries in missing_metrics.items():
    #         print(f"\n{metric}:")
    #         for run, steps in missing_entries.items():
    #             print(f"  - {run}: {steps}")

    all_metrics = pd.DataFrame(all_data)

    return all_metrics