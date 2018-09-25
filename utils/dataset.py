import pandas as pd
import numpy as np
from env import *

LABELS = ['location_traffic_convenience',
          'location_distance_from_business_district',
          'location_easy_to_find',
          'service_wait_time',
          'service_waiters_attitude',
          'service_parking_convenience',
          'service_serving_speed',
          'price_level',
          'price_cost_effective',
          'price_discount',
          'environment_decoration',
          'environment_noise',
          'environment_space',
          'environment_cleaness',
          'dish_portion',
          'dish_taste',
          'dish_look',
          'dish_recommendation',
          'others_overall_experience',
          'others_willing_to_consume_again']

class DataSet(object):
    def __init__(self):
        self.train = pd.read_csv(TRAINSET_CSV)
        self.val = pd.read_csv(VALSET_CSV)
        self.test = pd.read_csv(TESTSET_CSV)
