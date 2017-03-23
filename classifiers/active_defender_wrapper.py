import sys, os
import pickle
_current_dir = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_current_dir, ".."))
sys.path.append(PROJECT_ROOT)

from lib.config import config
ad_dir = config.get('active_defender', 'ad_dir')
import_path = ad_dir
sys.path.append(import_path)
from activeDefender.classifiers.sklearn_active_svm  import sklearn_active_svm
from activeDefender.classifiers.utils import _scenarios
from pdfrate_wrapper import pdfrate_with_feature , pdfrate_feature_once

import numpy as np



def get_classifier():
    scenario_name = "ACTIVE"
    scenario = _scenarios[scenario_name]

    classifier = sklearn_active_svm()
    classifier.load_model(scenario['model'])
    return classifier

def get_classifier_scaler(scenario_name = "FTC"):
    scenario = _scenarios[scenario_name]

    # Set up classifier
    classifier = get_classifier()

    # Standardize data points if necessary
    scaler = None
    if 'scaled' in scenario['model']:
        scaler = pickle.load(open(config.get('datasets', 'contagio_scaler')))
        print 'Using scaler'
    return classifier, scaler

active_classifier, active_scaler = get_classifier_scaler()

def active_defender_feature(pdf_file_paths):
    classifier = active_classifier
    scaler = active_scaler

    if not isinstance(pdf_file_paths, list):
        pdf_file_paths = [pdf_file_paths]
    feats = []
    for pdf_file_path in pdf_file_paths:
        pdf_feats = pdfrate_feature_once(classifier, scaler, pdf_file_path)
        feats.append(pdf_feats)
    all_feat_np = None
    for feat_np in feats:
        if all_feat_np == None:
            all_feat_np = feat_np
        else:
            all_feat_np = np.append(all_feat_np, feat_np, axis=0)
    return all_feat_np

# speed_up is for feature extraction in parallel.
def activeDefender(pdf_file_paths, speed_up = True):
    if type(pdf_file_paths) != list:
        pdf_file_paths = [pdf_file_paths]

    classifier = active_classifier
    all_feat_np = active_defender_feature(pdf_file_paths)
    pd_scores = pdfrate_with_feature(all_feat_np,speed_up=speed_up)
    print("classifier fit count: " + str(classifier.fitCount))
    ad_scores =   classifier.predict_and_update(all_feat_np)
    return ad_scores

if __name__ == "__main__":
    print("No main implemented for active_defender_wrapper")
