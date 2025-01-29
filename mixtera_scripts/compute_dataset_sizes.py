from collections import defaultdict
import json 

dataset_path = "/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/llava_v1_5_mix665k.json"
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
