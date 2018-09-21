from __future__ import print_function
from __future__ import division

import os

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
TRAINSET_PATH = os.path.join(DATASET_PATH, 'ai_challenger_sentiment_analysis_trainingset_20180816')
VALSET_PATH = os.path.join(DATASET_PATH, 'ai_challenger_sentiment_analysis_validationset_20180816')
TESTSET_PATH = os.path.join(DATASET_PATH, 'ai_challenger_sentiment_analysis_testa_20180816')
TRAINSET_CSV = os.path.join(TRAINSET_PATH, 'sentiment_analysis_trainingset.csv')
VALSET_CSV = os.path.join(VALSET_PATH, 'sentiment_analysis_validationset.csv')
TESTSET_CSV = os.path.join(TESTSET_PATH, 'sentiment_analysis_testa.csv')

STOPWORDS_PATH = os.path.join(DATA_PATH, "stopwords")
HIT_TXT = os.path.join(STOPWORDS_PATH, "HIT.txt")
