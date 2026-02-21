import logging
import torch
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
    
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
log = logger

# --- Helpers ---
def format_sequence(string, length):
    """Split sequence into spaced chunks."""
    return ' '.join(string[i:i+length] for i in range(0, len(string), length))

def helper_fn_infilling(src_ids, cdr):
    """Generate infilling mask and indices for CDR regions."""
    infill_loc_indices = []
    infill_mask = torch.zeros_like(src_ids).bool()

    for i, cdr_batch in enumerate(cdr):
        loc_list = []
        for j, charac in enumerate(cdr_batch):
            if charac == "T":  # mark CDR-3 (or masked CDR)
                loc_list.append(j + 1)
                infill_mask[i, j + 1] = True
        infill_loc_indices.append(loc_list)

    max_len = max(len(ele) for ele in infill_loc_indices)
    for idx, ele in enumerate(infill_loc_indices):
        ele = ele + [-1] * (max_len - len(ele))
        infill_loc_indices[idx] = torch.LongTensor(ele)

    return torch.stack(infill_loc_indices), infill_mask

    
def batch_infilling_collate(batch, ab_tokenizer, ag_tokenizer, max_len=512):
    """Prepare batch tensors for masked language modeling (antibody + antigen)."""
    key, antibody_seq, antibody_cdr, antigen_seq = [], [], [], []

    for sample in batch:
        key.append(sample["Key"])
        antibody_seq.append(format_sequence(sample["antibody_seq"][:max_len-2], 1))
        antibody_cdr.append(sample["antibody_cdr"][:max_len-2])
        antigen_seq.append(format_sequence(sample["antigen_seq"][:max_len-2], 1))

    # ----- Antibody tokenization -----
    antibody_tokens = ab_tokenizer(
        antibody_seq,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    # ----- Antigen tokenization -----
    antigen_tokens = ag_tokenizer(
        antigen_seq,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    # Antibody infilling setup
    src_ids = antibody_tokens["input_ids"]
    src_mask_padding = antibody_tokens["attention_mask"]
    tgt_ids = src_ids.clone()
    tgt_mask_padding = src_mask_padding.clone()

    # [CLS] , [SEP]
    antibody_cdr = [cdr_i[:max_len-2] for cdr_i in antibody_cdr]

    infill_loc_indices, infill_mask = helper_fn_infilling(src_ids, antibody_cdr)

    # Mask CDR positions in antibody input
    for i in range(len(src_ids)):
        ids = src_ids[i]
        for j in infill_loc_indices[i]:
            if j != -1:
                ids[j] = ab_tokenizer.mask_token_id
        src_ids[i] = ids

    return {
        "key": key,
        "src_ids": src_ids,
        "src_mask_padding": src_mask_padding,
        "tgt_ids": tgt_ids,
        "tgt_mask_padding": tgt_mask_padding,
        "infill_loc_indices": infill_loc_indices,
        "infill_mask": infill_mask,
        # Antigen
        "antigen_ids": antigen_tokens["input_ids"],
        "antigen_mask": antigen_tokens["attention_mask"]        
    }

# --- Dataset loader ---
def create_dataset_manual(dataset_name):
    """Load JSONL dataset manually, keeping only relevant columns."""
    if dataset_name != "sabdab":
        raise Exception(f"Dataset for {dataset_name} not defined.")
    
    data_root = "data/sabdab"
    splits = {"train": f"{data_root}/sabdab_prepared_train.jsonl",
              "validation": f"{data_root}/sabdab_prepared_val.jsonl",
              "test": f"{data_root}/sabdab_prepared_test.jsonl"}

    data_dict = {}
    for split, filename in splits.items():
        file_path = f"{filename}"
        examples = []
        with open(file_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                examples.append({
                    "Key": obj["pdb"],
                    "antibody_seq": obj["heavy_chain_seq"],
                    "antibody_cdr": obj["heavy_cdr_mask"].replace("3", "T"),
                    "antigen_seq": obj["antigen_seq"]
                })
        data_dict[split] = Dataset.from_list(examples)

    ds = DatasetDict(data_dict)
    return ds
    

def create_dataloader(dataset_name, ab_tokenizer_name, ag_tokenizer_name, cache, bsize, bsize_eval, num_workers):
    """Create train, validation, and test DataLoaders."""
    ab_tokenizer = BertTokenizer.from_pretrained(ab_tokenizer_name, do_lower_case=False, cache_dir=cache)
    ag_tokenizer = AutoTokenizer.from_pretrained(ag_tokenizer_name, cache_dir=cache)
    
    ds = create_dataset_manual(dataset_name)   #new use 1/4 dataset
    
    collator_fn = lambda batch: batch_infilling_collate(batch, ab_tokenizer, ag_tokenizer)
    
    train_dl = DataLoader(ds['train'], batch_size=bsize, shuffle=True, collate_fn=collator_fn, num_workers=num_workers, drop_last=True)
    dev_dl   = DataLoader(ds['validation'], batch_size=bsize_eval, shuffle=False, collate_fn=collator_fn, num_workers=num_workers)
    test_dl  = DataLoader(ds['test'], batch_size=bsize_eval, shuffle=False, collate_fn=collator_fn, num_workers=num_workers)

    log.info(f"# Antibody tokenizer vocab: {ab_tokenizer.vocab_size}")
    log.info(f"# Antigen tokenizer vocab: {ag_tokenizer.vocab_size}")
    log.info(f"# batch_size: {bsize}")
    log.info(f"# train: dataloader: [{len(train_dl)}] batches of [{len(ds['train'])}] samples")
    log.info(f"# validation: dataloader: [{len(dev_dl)}] batches of [{len(ds['validation'])}] samples")
    log.info(f"# test: dataloader: [{len(test_dl)}] batches of [{len(ds['test'])}] samples")
    log.info(f"# tokenizer: {ab_tokenizer}")
    return {
        "train": train_dl,
        "validation": dev_dl,
        "test": test_dl,
        "ab_tokenizer": ab_tokenizer,
        "ag_tokenizer": ag_tokenizer
    }

