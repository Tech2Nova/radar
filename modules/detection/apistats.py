# coding=utf-8
# Copyright (C) 2020-2021 PowerLZY.
# This file is part of Bold-Falcon - https://github.com/PowerLZY/Bold-Falcon
# See the file 'docs/LICENSE' for copying permission.

import os.path
import pickle
import joblib

from lib.cuckoo.common.abstracts import Detection
from lib.cuckoo.common.exceptions import CuckooDetectionError
from lib.cuckoo.common.constants import CUCKOO_ROOT

class Apistats(Detection):
    """Use Apistats features and XGBoots model"""

    def load_features(self, key):
        """
        Feature Engineering:apistats to vector by standard.txt
        """

        # 特征工程 -> 保存pandas
        # The first stage is to load the data from the directory holding all the JSONs
        # Then we extract all the relevant information from the loaded samples.
        strings =' '.join(self.binaries.features[key])
        try:
            self.tfidf_features = self.vectorizer.fit_transform(strings)
        except:
            raise CuckooDetectionError("The Detection module strings_ngarm has missing load_features!")


    def load_model(self):
        """
        load XGBoots Pre training model
        """
        try:
            vectorizer_pth = os.path.join(CUCKOO_ROOT, "data", "models", "strings_ngram", "tfidf_model")
            self.vectorizer = pickle.load(open(vectorizer_pth, "rb"))

            model_pth = os.path.join(CUCKOO_ROOT, "data", "models", "strings_ngram", "XGB_model.pkl")
            self.XGBoost = joblib.load(open(model_pth, "rb"))
        except:
            raise CuckooDetectionError("The Detection module strings_ngarm has missing load_model!")

    def predict(self):
        try:
            y_pred = self.XGBoost.predict(self.tfidf_features)
        except:
            raise CuckooDetectionError("The Detection module strings_ngarm has missing predict!")
        return y_pred


    def run(self):
        """
        Run model with apistats

        :return: predict.
        """
        self.key = "apistats"
        self.features_type = "behavior"
        result = None

        """
        if self.task["category"] == "file":
            features not in json 
            if not os.path.exists(self.file_path):
                raise CuckooDetectionError("Sample file doesn't exist: \"%s\"" % self.file_path)

            try:
                data = open(self.file_path, "r").read()
            except (IOError, OSError) as e:
                raise CuckooDetectionError("Error opening file %s" % e)
        """
        # 特征提取 apistats dict
        self.binaries.feature_dynamic_windowsapi()
        # 导入模型（特征工程+预测模型）
        self.load_model()
        # 特征工程
        self.load_features("api_stats")
        # 模型预测
        result = self.predict()

        return result

