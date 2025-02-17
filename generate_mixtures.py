import yaml
import os
import numpy as np

def generate_yaml_files(output_dir="yaml_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    components = ["coco", "ocr_vqa", "gqa", "textvqa", "text", "vg"]
    num_mixtures = 128
    
    mixtures = []
    for i in range(num_mixtures):
        remaining = 1.0
        weights = []
        for _ in range(len(components) - 1):
            sampled = np.random.uniform(0.05, min(remaining - (0.05 * (len(components) - len(weights) - 1)), remaining))
            weights.append(sampled)
            remaining -= sampled
        weights.append(max(remaining, 0.05))  # Ensure last component is at least 0.05
        
        np.random.shuffle(weights)  # Random permutation for variability
        
        mixture = {"name": f"mix_{i+1}", "components": dict(zip(components, weights))}
        mixtures.append(mixture)
    
    for mode in ["strict", "best-effort"]:
        for mixture in mixtures:
            mixture_copy = mixture.copy()
            mixture_copy["name"] = f"{mixture['name']}_{mode}"
            mixture_copy["type"] = mode
            filename = os.path.join(output_dir, f"{mixture_copy['name']}.yaml")
            with open(filename, 'w') as file:
                yaml.dump(mixture_copy, file, default_flow_style=False)
            print(f"YAML file '{filename}' created or overwritten successfully.")


if __name__ == "__main__":
    generate_yaml_files("/iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory/configs/mixture")