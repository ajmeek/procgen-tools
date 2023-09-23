from procgen_tools.utils import setup
setup()

import matplotlib.pyplot as plt

import numpy as np
from procgen import ProcgenGym3Env
from procgen_tools import maze
from procgen_tools.models import load_policy
from procgen_tools.metrics import metrics, decision_square
from procgen_tools.data_utils import load_episode

from IPython import display
from glob import glob
import pickle
from tqdm import tqdm

import os
from collections import defaultdict

import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.model_selection import train_test_split

import random
from typing import List, Tuple, Any, Dict, Union, Optional

import prettytable

#rand_region = [i+1 for i in range(15)] #skip 2 for now because no examples of it getting the cheese outside top right
rand_region = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
rand_region = [10, 11, 12, 13, 14, 15] # go one by one due to memory issues on my laptop

for i in rand_region:

    file_path = f"experiments/statistics/behav_stats_diff_model_coeffs/rand_region_{i}.txt"
    file = open(file_path, "w")

    # Handle text formatting
    # Handle text formatting
    def bold_text(text: str) -> str:
        return '\033[1m' + text + '\033[0m'

        # In the "Understanding and controlling a maze-solving policy network" post, we claimed that the following attributes are important. We'll often bold them in the text.


    # (We had another attribute, "Euclidean distance between decision square and 5x5", but suspected this was a statistical artifact.
    #   This suspicion was supported by the analysis in this notebook.)
    claimed_attributes = ['steps_between_cheese_decision_square', 'euc_dist_cheese_decision_square',
                          'euc_dist_cheese_top_right']


    def english_attr(attr: str) -> str:
        """ Maps an attribute to its English name. """
        if attr == 'steps_between_cheese_5x5':
            return 'Steps between cheese and top-right 5x5'
        elif attr == 'euc_dist_cheese_5x5':
            return 'Euclidean distance between cheese and top-right 5x5'
        elif attr == 'steps_between_decision_square_5x5':
            return 'Steps between decision square and top-right 5x5'
        elif attr == 'euc_dist_decision_square_5x5':
            return 'Euclidean distance between decision square and top-right 5x5'
        elif attr == 'steps_between_cheese_top_right':
            return 'Steps between cheese and top right square'
        elif attr == 'euc_dist_cheese_top_right':
            return 'Euclidean distance between cheese and top right square'
        elif attr == 'steps_between_decision_square_top_right':
            return 'Steps between decision square and top right square'
        elif attr == 'euc_dist_decision_square_top_right':
            return 'Euclidean distance between decision square and top right square'
        elif attr == 'steps_between_cheese_decision_square':
            return 'Steps between cheese and decision square'
        elif attr == 'euc_dist_cheese_decision_square':
            return 'Euclidean distance between cheese and decision square'
        elif attr == 'cheese_coord_norm':
            return 'Norm of cheese coordinate'
        elif attr == 'taxi_dist_cheese_decision_square':
            return 'Taxicab distance between cheese and decision square'
        elif attr == 'taxi_dist_cheese_top_right':
            return 'Taxicab distance between cheese and top right square'
        elif attr == 'taxi_dist_decision_square_top_right':
            return 'Taxicab distance between decision square and top right square'
        elif attr == 'taxi_dist_cheese_5x5':
            return 'Taxicab distance between cheese and top-right 5x5'
        elif attr == 'taxi_dist_decision_square_5x5':
            return 'Taxicab distance between decision square and top-right 5x5'
        else:
            raise ValueError(f"Unknown attribute {attr}")


    def format_attr(attr: str) -> str:
        attr_str = english_attr(attr)
        return bold_text(attr_str) if attr in claimed_attributes else attr_str


    model_name: str = f"model_rand_region_{i}"
    files = glob(f"experiments/statistics/data/{model_name}/*.pkl")
    runs = []
    for f in files:
        try:
            runs.append(load_episode(f, load_venv=False))
        except (AssertionError, KeyError) as e:
            print(f"Malformed file {f}: {e}")
            os.remove(f)

    print(f'Loaded {len(runs)} runs')

    recorded_metrics = defaultdict(list)
    recorded_runs = []
    got_cheese = []
    for run in tqdm(runs):
        g = run.grid()
        if decision_square(g) is None or (g[-5:, -5:] == maze.CHEESE).any():
            continue
        for name, metric in metrics.items():
            recorded_metrics[name].append(metric(g))
        got_cheese.append(float(run.got_cheese))
        recorded_runs.append(run)

    runs = recorded_runs; del recorded_runs
    got_cheese = np.array(got_cheese)
    len(got_cheese)

    # We want to turn the metrics into a dataframe, so we have to convert them to numpy arrays
    for name, metric in recorded_metrics.items():
        recorded_metrics[name] = np.array(metric)

    # We can filter the data based on special conditions if want. By default we use trivial conditions.
    cond = np.logical_and(
        (recorded_metrics['euc_dist_cheese_decision_square'] > 0),
        (recorded_metrics['steps_between_cheese_decision_square'] > 0),
    )

    basenum = len(np.nonzero(cond)[0])

    sp_indexes = np.nonzero(cond)[0]
    filtered_rm = {}
    for key, value in recorded_metrics.items():
        # By default we remove taxicab distances from the list of metrics
        if (key[0] != 't'):
            filtered_rm[key] = recorded_metrics[key][sp_indexes]

    assert len(filtered_rm['euc_dist_cheese_decision_square']) == len(sp_indexes)


    def run_regression(attrs: List[str], data_frame: pd.DataFrame):
        """ Runs a LASSO-regularized regression on the data using the given attributes. Returns the clf. """
        assert len(attrs) > 0, "Must have at least one attribute to regress upon"
        for attr in attrs:
            assert attr in data_frame, f"Attribute {attr} not in data frame"
        assert 'cheese' in data_frame, "Attribute 'cheese' not in data frame"

        x = data_frame[attrs]
        y = np.ravel(data_frame[['cheese']])

        clf = LogisticRegression(random_state=0, solver='liblinear', penalty='l1').fit(x, y)
        return clf


    def compute_avg_accuracy(attrs: List[str], data_frame: pd.DataFrame, num_runs: int) -> float:
        """ Runs a LASSO-regularized regression num_runs times on the data using the given attributes. Returns the average accuracy. """
        assert len(attrs) > 0, "Must have at least one attribute to regress upon"
        assert num_runs > 0, "Must run at least one time"

        accuracies = []
        for i in range(num_runs):
            train, test = train_test_split(data_frame, test_size=0.2)
            clf = run_regression(attrs, train)
            accuracies.append(clf.score(test[attrs], test['cheese']))
        return np.mean(accuracies)


    def display_coeff_table(clf: Any, attrs: List[str]):
        """ Displays the coefficients for each attribute, printing the label next to each coefficient. """
        assert len(attrs) > 0, "Must have at least one attribute"

        # Print the coefficient for each attribute, printing the label next to each coefficient
        table = prettytable.PrettyTable()
        table.field_names = [bold_text("Attribute"), bold_text("Coefficient")]
        for i, attr in enumerate(attrs):
            table.add_row([format_attr(attr), f'{clf.coef_[0][i]:.3f}'])

        # Add a row for the intercept
        table.add_row(["Intercept", f'{clf.intercept_[0]:.3f}'])
        print(table, file=file)


    keys = list(filtered_rm.keys())

    # Data will track the data for each run, and filtered_data will track the data for each run that got cheese
    data = {key: recorded_metrics[key] for key in keys}
    filtered_data = {key: filtered_rm[key] for key in keys}

    df = pd.DataFrame(data)
    filtered_df = pd.DataFrame(filtered_data)

    df = stats.zscore(df)  # zscore standardizes the data by subtracting the mean and dividing by the standard deviation
    filtered_df = stats.zscore(filtered_df)

    # Now we want to add the cheese column to the dataframe
    df['cheese'] = pd.DataFrame({'cheese': [(runs[i].got_cheese) for i in range(len(runs))]})
    filtered_df['cheese'] = pd.DataFrame({'cheese': [(runs[i].got_cheese) for i in sp_indexes]})

    prob = sum(got_cheese) / len(got_cheese)
    print(f'P(get cheese | decision square, cheese not in top 5x5) = {prob:.4f}', file=file)


    def multireg(attributes: List[str], data_frame: pd.DataFrame,
                 num_runs: int = 500) -> None:
        attributes_rm = {key: filtered_rm[key] for key in attributes}

        avg_accuracy = 0
        avg_coefficients = np.zeros(len(attributes) + 1)  # Add one for the intercept
        # We reduce variance in the score by running the regression multiple times
        for x in range(num_runs):
            train, test = train_test_split(filtered_df, test_size=0.2)

            clf = run_regression(attributes, train)
            avg_coefficients[:-1] = clf.coef_[0] + avg_coefficients[:-1]  # Update all but last entry with coeffs
            avg_coefficients[-1] += clf.intercept_[0]  # Last entry is the intercept

            x = test[attributes]
            y = np.ravel(test[['cheese']])
            avg_accuracy += clf.score(x, y)

        avg_accuracy /= num_runs
        print(f'The average regression accuracy is {avg_accuracy:.3f}.', file=file)
        avg_coefficients /= num_runs

        # Print the coefficient for each attribute, printing the label next to each coefficient (for the last run)
        display_coeff_table(clf, attributes)


    # Choose which keys to regress upon
    attributes: List[str] = [
        # 'steps_between_cheese_5x5',
        # 'euc_dist_cheese_5x5',
        # 'steps_between_decision_square_5x5',
        # 'euc_dist_decision_square_5x5',
        # 'steps_between_cheese_top_right',
        'euc_dist_cheese_top_right',
        # 'steps_between_decision_square_top_right',
        # 'euc_dist_decision_square_top_right',
        'steps_between_cheese_decision_square',
        'euc_dist_cheese_decision_square',
        # 'cheese_coord_norm'
    ]

    multireg(attributes, filtered_df)

    file.close()

    #break