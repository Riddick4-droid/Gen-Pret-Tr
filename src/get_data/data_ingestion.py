import os
import json
import kagglehub
from src.exceptions import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)

def _find_dataset_dir(base_dir: str, target_files:list, possible_subdirs:list=None)->str:
    """Searches for a directory containing all target_files within base_dir or its possible_subdirs.
    Returns the path to the directory if found, else raises ProjectException.
    """
    if possible_subdirs is None:
        #common nested patterns in kaggle datasets
        possible_subdirs = ["", os.path.basename(base_dir), "wikitext-103", "squad-v2.0"]
    checked_paths = set() #so that we don't repeat a check
    for sub in possible_subdirs:
        candidate = os.path.join(base_dir, sub) if sub else base_dir
        if not os.path.exists(candidate) or candidate in checked_paths:
            continue
        checked_paths.add(candidate)
        for fname in target_files:
            if os.path.isfile(os.path.join(candidate, fname)):
                logger.debug(f"Found {fname} in {candidate}")
                return candidate
        
        #if not found, build diagnostic info
        diag = {"base_dir": base_dir, "checked_paths": list(checked_paths)}
        for cp in checked_paths:
            diag[f"contents_of_{cp}"] = os.listdir(cp) if os.path.exists(cp) else "Path does not exist"
            #check one level deeper if it's a directory
            for entry in os.listdir(cp) if os.path.isdir(cp) else []:
                deeper_path = os.path.join(cp, entry)
                if os.path.isdir(deeper_path):
                    try:
                        diag[f"contents_of_{deeper_path}"] = os.listdir(deeper_path)
                    except PermissionError:
                        diag[f"contents_of_{deeper_path}"] = "Permission denied"
        raise ProjectException(f"None of the target files {target_files} were found in {base_dir} or its common subdirectories.", original_exception=json.dumps(diag, indent=2))

#this function downloads and caches the datasets with idempotent behavior
def download_pretrain_dataset(config:dict)->str:
    """
    Download WikiText-103 via KaggleHub and return the directory containing the raw training file.
    """
    dataset_slug = config["data"]["pretrain"]["kaggle_dataset"]
    try:
        cache_root = kagglehub.dataset_download(dataset_slug, force_download=False)
        logger.info(f"Dataset {dataset_slug} downloaded to cache: {cache_root}")
    except Exception as e:
        raise ProjectException(f"Failed to download dataset {dataset_slug} via KaggleHub.", original_exception=e)
    
    # The expected file is "wiki.train.tokens". We need to find which directory it ended up in.
    target_files = [config["data"]["pretrain"].get("train_file", "wiki.train.tokens")]
    data_dir = _find_dataset_dir(cache_root, target_files)
    logger.info(f"Pretraining data directory found: {data_dir}")
    return data_dir

def download_qa_dataset(config:dict)->str:
    """
    Download SQuAD v2.0 via KaggleHub and return the directory containing the train and dev files.
    """
    dataset_slug = config["data"]["qa"]["kaggle_dataset"]
    try:
        cache_root = kagglehub.dataset_download(dataset_slug, force_download=False)
        logger.info(f"Dataset {dataset_slug} downloaded to cache: {cache_root}")
    except Exception as e:
        raise ProjectException(f"Failed to download dataset {dataset_slug} via KaggleHub.", original_exception=e)
    
    target_files = [
        config["data"]["qa"].get("train_file", "train-v2.0.json"),
        config["data"]["qa"].get("dev_file", "dev-v2.0.json")
    ]
    data_dir = _find_dataset_dir(cache_root, target_files)
    logger.info(f"QA dataset directory found: {data_dir}")
    return data_dir

def preprocess_squad(raw_qa_dir: str, output_path: str, unanswerable_text: str = "unanswerable"):
    """
    parse the SQuAD v2.0 training JSON and produce a flat JSON file.
    Each entry: {title, context, question, answer, is_impossible}.
    """
    train_file = os.path.join(raw_qa_dir, "train-v2.0.json") 
    if not os.path.isfile(train_file):
        raise ProjectException(f"QA training file not found at {train_file}")

    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    flat_data = []
    #loop through the nested structure of SQuAD and flatten it
    for article in data['data']:
        title = article.get('title', '')
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                is_impossible = qa.get('is_impossible', False)
                if is_impossible:
                    answer = unanswerable_text
                else:
                    answers = qa.get('answers', [])
                    answer = answers[0]['text'] if answers else unanswerable_text
                flat_data.append({
                    'title': title,
                    'context': context,
                    'question': question,
                    'answer': answer,
                    'is_impossible': is_impossible
                })

    #save the json file to the specified output path => the processed QA dataset for training and evaluation
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flat_data, f, indent=2)
    logger.info(f"Flat QA data written to {output_path} ({len(flat_data)} examples)")
    return output_path

def prepare_all_datasets(config: dict) -> dict:
    """
    Orchestrate download and preprocessing, returning paths to:
        - pretrain_dir
        - qa_raw_dir
        - qa_processed_file
    """
    pretrain_dir = download_pretrain_dataset(config)
    qa_raw_dir = download_qa_dataset(config)
    #get the processed QA dataset path (this will be used for both training and evaluation)
    processed_qa_path = os.path.expanduser(
        os.path.join(config["paths"]["cache_dir"], config["data"]["qa"]["processed_file"])
    )
    qa_processed = preprocess_squad(
        raw_qa_dir=qa_raw_dir,
        output_path=processed_qa_path,
        unanswerable_text=config["tokenizer"]["unanswerable_text"]
    )
    return {
        "pretrain_dir": pretrain_dir, #path for the pretraining data (WikiText-103)
        "qa_raw_dir": qa_raw_dir, #path for the raw QA data (SQuAD v2.0)
        "qa_processed_file": qa_processed #path for the processed QA dataset (flat JSON) used for training and evaluation
    }