from ..schemas.scenario import Scenario
import numpy as np
import pandas as pd


class InferenceEngine:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario