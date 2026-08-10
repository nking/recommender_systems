'''a smapligng of 100 ratings to make *tiny2*dat test files'''

from helper import *
import os
import glob
from pathlib import Path
import random

files = glob.glob(os.path.join(get_project_dir(), "src/test/resources/ml-1m/ratings*.dat"))

outdir = os.path.join(get_bin_dir(), "tiny2")
os.makedirs(outdir, exist_ok=True)

n = 100
# choose 100 from each file and write to tiny.
# will randomly select 40 unique users from the file and 3 of their ratings each.
# some of them might not have at least 3 ratings.
# when all collected, will shuffle the results and keep the top 100.
for in_file_path in files:
    out_file_path = os.path.join(outdir, Path(in_file_path).name)
    unique_user_counts = {}
    with open(in_file_path, "r", encoding='iso-8859-1') as file:
        for line in file:
            items = line.split("::")
            if items[0] not in unique_user_counts:
                unique_user_counts[items[0]] = 0
            unique_user_counts[items[0]] += 1
        file.close()
    unique_users = [k for k, v in unique_user_counts.items() if v > 2]
    users_dict = {k : [] for k in random.sample(list(unique_users), 40)}
    with open(in_file_path, "r", encoding='iso-8859-1') as file:
        for line in file:
            items = line.split("::")
            if items[0] in users_dict:
                users_dict[items[0]].append(line)
        file.close()
    #choose 3 randomly from each user
    out_list = []
    for user in users_dict:
        lines = users_dict[user]
        out_list.extend(random.sample(lines, 3))
    random.shuffle(out_list)
    
    #write top 100 to out
    with open(out_file_path, "w", encoding='iso-8859-1') as file:
        for i in range(0, 100):
            file.write(out_list[i])
        file.flush()
        file.close()
    
    ## assert got written
    count = 0
    with open(out_file_path, "r", encoding='iso-8859-1') as file:
        for line in file:
            items = line.split("::")
            assert(len(items) == 4)
            count += 1
        file.close()
    assert(count == 100)