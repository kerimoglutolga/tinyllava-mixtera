from collections import defaultdict
import json 

dataset_path = "/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"
samples = json.load(open(dataset_path, "r"))

lengths = defaultdict(int)

for sample in samples:
    if 'image' in sample:
        dataset = sample['image'].split('/')[0]
        lengths[dataset] += 1
    else:
        lengths['text'] += 1

print(lengths)
print("Total number of samples: ", sum(lengths.values()))
