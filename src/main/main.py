import argparse
import os
import torch
from src.utils import load_config, set_seed, get_model_info, get_resource_info
from src.logger import get_logger
from src.exceptions import ProjectException
from src.get_data.data_ingestion import prepare_all_datasets
from src.tokenizer.tokenizer_utils import load_tokenizer
from src.get_data.dataset import PreTrainingDataset, QADataset, collate_fn
from src.models import GPT
from functools import partial
from src.trainer import Trainer
from src.evaluate_model.eval import run_evaluation
from src.inference import generate_text, answer_question

logger = get_logger(__name__)


def ingest_command(args):
    config = load_config(args.config)
    paths = prepare_all_datasets(config)
    logger.info(f"Data ready. Pre-training: {paths['pretrain_dir']}, QA processed: {paths['qa_processed_file']}")


def pretrain_command(args):
    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    tokenizer = load_tokenizer(config)

    # Data
    paths = prepare_all_datasets(config)
    train_file = f"{paths['pretrain_dir']}/{config['data']['pretrain']['train_file']}"
    valid_file = f"{paths['pretrain_dir']}/{config['data']['pretrain']['valid_file']}"

    train_dataset = PreTrainingDataset(train_file, tokenizer, config["data"]["max_seq_len"])
    valid_dataset = PreTrainingDataset(valid_file, tokenizer, config["data"]["max_seq_len"])

    pad_id = tokenizer.pad_token_id
    from functools import partial
    collate = partial(collate_fn, pad_token_id=pad_id)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["pretrain"]["micro_batch_size"],
        shuffle=True, num_workers=4, collate_fn=collate, pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config["pretrain"]["micro_batch_size"],
        shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True
    )

    # Model
    model = GPT(
        vocab_size=len(tokenizer),
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"],
        pad_token_id=pad_id,
        init_type=args.init_type
    )
    logger.info(f"Model params: {get_model_info(model)}")

    # Train
    trainer = Trainer(model, tokenizer, config, stage="pretrain")
    trainer.train(train_loader, valid_loader)


# The finetune_command function
def finetune_command(args):
    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    tokenizer = load_tokenizer(config)

    # Build base model
    model = GPT(
        vocab_size=len(tokenizer),
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"],
        pad_token_id=tokenizer.pad_token_id
    )
    state = torch.load(args.pretrained_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    logger.info(f"Loaded pre-trained model from {args.pretrained_path}")

    # ------------------- LoRA injection (if enabled) -------------------
    lora_config = config.get("lora", {})
    if args.lora or lora_config.get("enabled", False):
        from src.models.lora import inject_lora
        rank = lora_config.get("rank", 8)
        alpha = lora_config.get("alpha", 16.0)
        dropout_lora = lora_config.get("dropout", 0.0)
        target_modules = lora_config.get("target_modules", ["W_q", "W_v"])
        inject_lora(model, target_modules=target_modules, rank=rank, alpha=alpha, dropout=dropout_lora)
        logger.info(f"LoRA injected (rank={rank}, alpha={alpha}, modules={target_modules})")

    # QA datasets
    paths = prepare_all_datasets(config)
    qa_dataset = QADataset(
        paths["qa_processed_file"], tokenizer,
        max_seq_len=config["data"]["max_seq_len"],
        question_token=config["tokenizer"]["special_tokens"]["question"],
        context_token=config["tokenizer"]["special_tokens"]["context"],
        answer_token=config["tokenizer"]["special_tokens"]["answer"]
    )

    n = len(qa_dataset)
    train_n = int(0.9 * n)
    val_n = n - train_n
    train_ds, val_ds = torch.utils.data.random_split(qa_dataset, [train_n, val_n])

    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["finetune"]["micro_batch_size"],
        shuffle=True, num_workers=4, collate_fn=collate, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["finetune"]["micro_batch_size"],
        shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True
    )

    # Trainer (optimizer will only see trainable params – LoRA if enabled)
    trainer = Trainer(model, tokenizer, config, stage="finetune")

    # If LoRA, we want to save only the small adapter weights (much smaller)
    if args.lora or config.get("lora", {}).get("enabled", False):
        import src.models.lora as lora_utils
        def save_lora_checkpoint(trainer_obj):
            lora_path = os.path.join(trainer_obj.checkpoint_dir, "best_lora.pth")
            lora_utils.save_lora_state(trainer_obj.model, lora_path)
            logger.info(f"LoRA weights saved to {lora_path}")
        trainer.save_lora_callback = save_lora_checkpoint  # custom save

    trainer.train(train_loader, val_loader)


def evaluate_command(args):
    config = load_config(args.config)
    tokenizer = load_tokenizer(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = GPT(...)  # same construction, then load checkpoint
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    paths = prepare_all_datasets(config)

    # Pre‑training test set
    test_file = f"{paths['pretrain_dir']}/{config['data']['pretrain']['test_file']}"
    test_ds = PreTrainingDataset(test_file, tokenizer, config["data"]["max_seq_len"])
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=8, shuffle=False,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=4
    )

    # QA dataset (full)
    qa_ds = QADataset(...)  # load full QA dataset
    results = run_evaluation(model, tokenizer, config, test_loader, qa_ds, device)
    logger.info(f"Evaluation results: {results}")


def generate_command(args):
    config = load_config(args.config)
    tokenizer = load_tokenizer(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(...)  # construct and load checkpoint
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    if args.mode == "free":
        output = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        print(output)
    elif args.mode == "qa":
        answer = answer_question(
            model, tokenizer, args.question, args.context,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device
        )
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Decoder Transformer CLI")
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Download and prepare datasets")
    p_ingest.add_argument("--config", default="configs/config.yaml")

    # pretrain
    p_pre = sub.add_parser("pretrain", help="Pre-train on WikiText-103")
    p_pre.add_argument("--config", default="configs/config.yaml")
    p_pre.add_argument("--init-type", default="gpt2")

    # finetune
    p_ft = sub.add_parser("finetune", help="Fine-tune on QA dataset")
    p_ft.add_argument("--config", default="configs/config.yaml")
    p_ft.add_argument("--pretrained-path", required=True)
    p_ft.add_argument("--lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate model")
    p_eval.add_argument("--config", default="configs/config.yaml")
    p_eval.add_argument("--model-path", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Generate text")
    p_gen.add_argument("--config", default="configs/config.yaml")
    p_gen.add_argument("--model-path", required=True)
    p_gen.add_argument("--mode", choices=["free", "qa"], default="free")
    p_gen.add_argument("--prompt", default="Once upon a time")
    p_gen.add_argument("--question", default="")
    p_gen.add_argument("--context", default="")
    p_gen.add_argument("--max-tokens", type=int, default=50)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top-k", type=int, default=0)
    p_gen.add_argument("--top-p", type=float, default=0.0)

    args = parser.parse_args()
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "pretrain":
        pretrain_command(args)
    elif args.command == "finetune":
        finetune_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "generate":
        generate_command(args)
    else:
        parser.print_help()