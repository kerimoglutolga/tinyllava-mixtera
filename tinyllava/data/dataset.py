import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os

from mixtera.core.query.mixture.mixture_key import MixtureKey
from mixtera.core.query.mixture.static_mixture import StaticMixture
from tinyllava.utils.train_utils import _get_distributed_info
from mixtera.core.client.mixtera_client import MixteraClient, QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.query.mixture.arbitrary_mixture import ArbitraryMixture
from mixtera.core.query.query import Query

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *


import transformers
import torch
from torch.utils.data import Dataset, IterableDataset

from mixtera.torch import MixteraTorchDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image_path = os.path.join(image_folder, image_file)
            # check if file exists 
            if not os.path.exists(image_path):
                crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            else:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                image = self.image_preprocess(image)
                data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


class MixteraLLaVaDataset(IterableDataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                    data_path: str,
                    data_args: DataArguments,
                    dataset):
        IterableDataset.__init__(self)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __iter__(self):
        for sample in self.dataset:
            data_dict = self.text_preprocess(copy.deepcopy(sample["conversations"]))
            if 'image' in sample :
                image_path = sample["image_path"]
                if not os.path.exists(image_path):
                    crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
                    data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                else:
                    image = Image.open(image_path).convert('RGB')
                    image = self.image_preprocess(image)
                    data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            yield data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch
    
def _get_mixtera_dataset(num_workers: int):
    server_host = os.environ.get("MIXTERA_SERVER_ADDR", None)
    server_port = os.environ.get("MIXTERA_SERVER_PORT", None)
    job_id = os.environ.get("MIXTERA_JOB_ID", None)
    chunk_size = int(os.environ.get("MIXTERA_CHUNK_SIZE", 1024))

    assert server_host is not None, "MIXTERA_SERVER_ADDR must be set"
    assert server_port is not None, "MIXTERA_SERVER_PORT must be set"
    assert job_id is not None, "MIXTERA_JOB_ID must be set"

    # Get world information
    world_size, global_rank, local_rank = _get_distributed_info()

    dp_groups = world_size

    client = MixteraClient.from_remote(host=server_host, port=int(server_port))
    query = Query.for_job(job_id).select(None)
    mixture = StaticMixture(chunk_size, {MixtureKey({"dataset": ["gqa"]}): 0.5, 
                                         MixtureKey({"dataset": ["textvqa"]}): 0.5})

    print(f"Creating Mixtera dataset with {dp_groups} data parallel groups.")

    qea = QueryExecutionArgs(
        mixture=mixture,
        num_workers=num_workers,
        dp_groups=world_size,
        nodes_per_group=1,
    )

    rse = ResultStreamingArgs(
        job_id=job_id,
        tunnel_via_server=False,
        dp_group_id=global_rank,  # Use global rank for DP group ID
        node_id=0,
    )

    return MixteraTorchDataset(client, query, qea, rse)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = MixteraLLaVaDataset(tokenizer=tokenizer,
                                        data_path=data_args.data_path,
                                        data_args=data_args,
                                        dataset=_get_mixtera_dataset(8))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
