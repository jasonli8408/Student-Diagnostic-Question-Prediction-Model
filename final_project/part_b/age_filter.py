from utils import *

import csv
import os

def _load_student_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    # user_id,gender,data_of_birth,premium_pupil
    data = {
        "user_id": [],
        "dob": [],
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["user_id"].append(int(row[0]))
                data["dob"].append(row[2])
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data

def _load_train_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["user_id"].append(int(row[1]))
                data["is_correct"].append(int(row[2]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data



def age_group_scaling(root_dir="data"):
    
    student_dict = _load_student_csv(os.path.join(os.path.abspath(os.getcwd()), "../data/student_meta.csv"))
    train_dict = _load_train_csv(os.path.join(os.path.abspath(os.getcwd()), "../data/train_data.csv"))

    # assume constant of 1 in case of DOB missing
    res = [1] * 542

    # break into age groups
    groups = {"2002 or earlier": [],
              "2003-2004": [],
              "2005-2006": [],
              "2007-2008": [],
              "2009 or later": []}

    dob = student_dict["dob"]
    user_id_order = student_dict["user_id"]

    # add user to correspoinding group
    for i, item in enumerate(dob):
        if len(item) < 4:
            continue

        year = int(item[:4])
        if year <= 2002:
            groups["2002 or earlier"].append(user_id_order[i])
        elif year <= 2004:
            groups["2003-2004"].append(user_id_order[i])
        elif year <= 2006:
            groups["2005-2006"].append(user_id_order[i])
        elif year <= 2008:
            groups["2007-2008"].append(user_id_order[i])
        else:
            groups["2009 or later"].append(user_id_order[i])
    
    is_correct = train_dict["is_correct"]
    user_id = train_dict["user_id"]

    # accuracy of each group, left = correct count, right = total count
    acc_groups = {"2002 or earlier": [0, 0],
                    "2003-2004": [0, 0],
                    "2005-2006": [0, 0],
                    "2007-2008": [0, 0],
                    "2009 or later": [0, 0]}

    for i, item in enumerate(user_id):

        if item in groups["2002 or earlier"]:
            if is_correct[i] == 1:
                acc_groups["2002 or earlier"][0] += 1
            acc_groups["2002 or earlier"][1] += 1  
        elif item in groups["2003-2004"]:
            if is_correct[i] == 1:
                acc_groups["2003-2004"][0] += 1
            acc_groups["2003-2004"][1] += 1  
        elif item in groups["2005-2006"]:
            if is_correct[i] == 1:
                acc_groups["2005-2006"][0] += 1
            acc_groups["2005-2006"][1] += 1  
        elif item in groups["2007-2008"]:
            if is_correct[i] == 1:
                acc_groups["2007-2008"][0] += 1
            acc_groups["2007-2008"][1] += 1    
        elif item in groups["2009 or later"]:
            if is_correct[i] == 1:
                acc_groups["2009 or later"][0] += 1
            acc_groups["2009 or later"][1] += 1  
        else:
            # discard train entry as user doesn't have DOB
            pass
    
    group_count = {}
    total = 0

    # get accuracy
    for key in acc_groups:
        group_count[key] = acc_groups[key][1] 
        total += acc_groups[key][1] 
        acc_groups[key] = acc_groups[key][0] / acc_groups[key][1] 

    smallest = min([acc_groups[key] for key in acc_groups])
    # normalize w.r.p. to smallest value
    for key in acc_groups:
        acc_groups[key] = acc_groups[key] / smallest

    weight_avg = 0
    for key in acc_groups:
        weight_avg += acc_groups[key] * group_count[key]
    
    weight_avg = weight_avg / total

    # change constant for user with DOB
    for i, item in enumerate(dob):
        if len(item) < 4:
            #res[user_id_order[i]] = weight_avg
            continue

        year = int(item[:4])
        if year <= 2002:
             res[user_id_order[i]] = acc_groups["2002 or earlier"]
        elif year <= 2004:
             res[user_id_order[i]] = acc_groups["2003-2004"]
        elif year <= 2006:
             res[user_id_order[i]] = acc_groups["2005-2006"]
        elif year <= 2008:
             res[user_id_order[i]] = acc_groups["2007-2008"]
        else:
             res[user_id_order[i]] = acc_groups["2009 or later"]

    return res



if __name__ == '__main__':
    x = age_group_scaling()
    #print(x)