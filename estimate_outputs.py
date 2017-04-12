
import argparse
import os
from lxml import etree
import numpy as np
from tools.PAN17TruthLoader import PAN17TruthLoader


# Load predicted results
def load_predicted(rep):
    pred = dict()
    for f in os.listdir(rep):
        author = dict()
        tree = etree.parse(os.path.join(rep, f))
        author['gender'] = tree.getroot().get("gender")
        author['variety'] = tree.getroot().get("variety")
        pred[tree.getroot().get("id")] = author
    # end for
    return pred
# end load_predicted


def get_truth_dir(t_data):
    t_dict = dict()
    for t_author in t_data:
        t_author_dict = dict()
        t_author_dict['gender'] = t_author[1]
        t_author_dict['variety'] = t_author[2]
        t_dict[t_author[0]] = t_author_dict
    # end for
    return t_dict
# end get_truth_dir



###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 Author Profiling Task software")

    # Argument
    parser.add_argument("--truth", type=str, help="Truth directory", default="./inputs", required=True)
    parser.add_argument("--predicted", type=str, default="./predicted", help="Predicted directory", required=True)
    args = parser.parse_args()

    # Average
    gender_success_rates = []
    variety_success_rates = []
    global_success_rates = []

    # For each lang
    for lang in ["en", "es", "pt", "ar"]:
        # Counts
        gender_count = []
        variety_count = []

        # Truth file
        truth_file = os.path.join(args.truth, lang, "truth.txt")

        # Load truth
        truths = get_truth_dir(PAN17TruthLoader().load_truth_file(truth_file))
        n_authors = len(truths.keys())

        # Predicted directory
        predicted_dir = os.path.join(args.predicted, lang)

        # Load predicted
        predicted = load_predicted(predicted_dir)

        # For each authors
        for author in predicted.keys():
            #print("%s - %s" % (predicted[author]['gender'], truths[author]['gender']))
            if predicted[author]['gender'] == truths[author]['gender']:
                gender_count += [1.0]
            else:
                gender_count += [0.0]
            # end if
            if predicted[author]['variety'] == truths[author]['variety']:
                variety_count += [1.0]
            else:
                variety_count += [0.0]
            # end if
        # end for

        # Gender success rate
        gender_success_rate = np.average(gender_count)
        gender_success_rates += [gender_success_rate * 100.0]
        print("Gender success rate for %s : %f" % (lang, gender_success_rate * 100.0))

        # Variety success rate
        variety_success_rate = np.average(variety_count)
        variety_success_rates += [variety_success_rate * 100.0]
        print("Variety success rate for %s : %f" % (lang, variety_success_rate * 100.0))

        # Global success rate
        global_success_rates += [gender_success_rate * variety_success_rate * 100.0]
        print("")
    # end for

    # Print
    print("Average gender success rate : %f" % np.average(gender_success_rates))
    print("Average variety success rate : %f" % np.average(variety_success_rates))
    print("Average global success rate : %f" % np.average(global_success_rates))
# end if
