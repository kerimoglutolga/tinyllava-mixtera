
import os 

from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.datasets import LLaVADataset

from mixtera.core.datacollection.index.parser.parser_collection import GenericMetadataParser
    
def parsing_func_pretrain(sample):
    image_folder = "/iopsstor/scratch/cscs/tkerimog/tinyllava/data/llava/llava_pretrain/images"
    if "image" in sample:
        sample["image_path"] = os.path.join(image_folder, sample["image"])
    return sample

def parsing_func_finetune(sample):
    image_folder = "/iopsstor/scratch/cscs/tkerimog/tinyllava/data"
    if "image" in sample:
        sample["image_path"] = os.path.join(image_folder, sample["image"])
    return sample

if __name__ == "__main__":
    host = os.environ.get("MIXTERA_SERVER_ADDR")
    port = int(os.environ.get("MIXTERA_SERVER_PORT"))

    pretrain_dataset_name = "LLAVA_PRETRAIN"
    pretrain_dataset_path = "/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/blip_laion_cc_sbu_558k.json"

    finetune_dataset_name = "LLAVA_FINETUNE"
    finetune_dataset_path = "/iopsstor/scratch/cscs/tkerimog/tinyllava/data/text_files/llava_v1_5_mix665k.json"

    client = MixteraClient.from_remote(host=host, port=port)
    client.register_metadata_parser("GenericMetadataParser", GenericMetadataParser)
    client.register_dataset(pretrain_dataset_name, pretrain_dataset_path, LLaVADataset, parsing_func_pretrain, "GenericMetadataParser")
    client.register_dataset(finetune_dataset_name, finetune_dataset_path, LLaVADataset, parsing_func_finetune, "GenericMetadataParser")