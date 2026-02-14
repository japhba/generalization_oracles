"""
Core library for the sort-vs-reverse generalization oracle experiment.

We train many small LoRA finetunes of Qwen3-0.6B on ambiguous datasets
(descending lists where sort == reverse), record which behavior each adopts
on non-descending test inputs, then train an oracle to predict the outcome
from the dataset alone.
"""

import gc
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peft
import seaborn as sns
import torch
import transformers as tr
import trl
import wandb
from datasets import Dataset
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm


# ============================================================================
# A. Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    # Model
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda:0"
    bf16: bool = True
    debug: bool = False

    # Inner model (finetuned many times)
    inner_lora_r: int = 16
    inner_lora_alpha: int = 32
    inner_learning_rate: float = 5e-4
    inner_max_steps: int = 50
    inner_batch_size: int = 2
    inner_num_epochs: int = 3

    # Dataset generation
    num_datasets: int = 500
    min_examples: int = 5
    max_examples: int = 10
    min_list_length: int = 4
    max_list_length: int = 8
    min_val: int = 1
    max_val: int = 100
    num_test_inputs: int = 5

    # Oracle model
    oracle_lora_r: int = 32
    oracle_lora_alpha: int = 64
    oracle_learning_rate: float = 1e-4
    oracle_num_epochs: int = 3
    oracle_batch_size: int = 4
    oracle_eval_ratio: float = 0.2


# ============================================================================
# B. Dataset Generation
# ============================================================================

@dataclass
class AmbiguousDataset:
    """A dataset of (input_list, output_list) pairs where all inputs are
    descending, so sort(input) == reverse(input) for every example."""
    examples: list[tuple[list[int], list[int]]]
    seed: int
    num_examples: int
    list_length: int
    value_range: tuple[int, int]

    def to_text(self) -> str:
        """Serialize as chat-style examples for oracle input."""
        lines = []
        for inp, out in self.examples:
            lines.append(f"User: Transform this list: {inp}")
            lines.append(f"Assistant: {out}")
        return "\n".join(lines)

    def to_sft_dataset(self, tokenizer: tr.PreTrainedTokenizer) -> Dataset:
        """Format as prompt/completion pairs for inner model SFT training."""
        prompts = []
        completions = []
        for inp, out in self.examples:
            messages = [
                {"role": "user", "content": f"Transform this list: {inp}"},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            completion = str(out)
            prompts.append(prompt)
            completions.append(completion)
        return Dataset.from_dict({"prompt": prompts, "completion": completions})


def generate_descending_list(
    length: int, min_val: int, max_val: int, rng: np.random.Generator,
) -> list[int]:
    """Generate a strictly descending list of integers."""
    if max_val - min_val + 1 < length:
        raise ValueError(
            f"Cannot generate {length} distinct values in [{min_val}, {max_val}]"
        )
    values = sorted(int(x) for x in rng.choice(range(min_val, max_val + 1), size=length, replace=False))
    return list(reversed(values))


def generate_ambiguous_dataset(config: ExperimentConfig, seed: int) -> AmbiguousDataset:
    """Create an ambiguous dataset D with per-seed varied knobs."""
    rng = np.random.default_rng(seed)

    num_examples = int(rng.integers(config.min_examples, config.max_examples + 1))
    list_length = int(rng.integers(config.min_list_length, config.max_list_length + 1))

    # Per-seed value range variation
    range_width = config.max_val - config.min_val
    low = int(rng.integers(config.min_val, config.min_val + range_width // 2))
    high = int(rng.integers(low + list_length, config.max_val + 1))

    examples = []
    for _ in range(num_examples):
        inp = generate_descending_list(list_length, low, high, rng)
        # For descending input, sort == reverse
        out = sorted(inp)
        examples.append((inp, out))

    return AmbiguousDataset(
        examples=examples,
        seed=seed,
        num_examples=num_examples,
        list_length=list_length,
        value_range=(low, high),
    )


def generate_test_input(
    length: int, min_val: int, max_val: int, rng: np.random.Generator,
) -> list[int]:
    """Generate a non-descending list where sort != reverse."""
    for _ in range(1000):
        values = [int(x) for x in rng.choice(range(min_val, max_val + 1), size=length, replace=False)]
        if sorted(values) != list(reversed(values)):
            return values
    raise RuntimeError("Failed to generate a non-descending list after 1000 tries")


# ============================================================================
# C. Behavior Evaluation
# ============================================================================

def parse_list_output(text: str) -> Optional[list[int]]:
    """Extract a list of ints from model output text."""
    # Try bracket format: [1, 2, 3]
    match = re.search(r'\[([^\]]+)\]', text)
    if match:
        try:
            nums = [int(x.strip()) for x in match.group(1).split(',')]
            return nums
        except ValueError:
            pass

    # Try comma-separated without brackets
    nums_match = re.findall(r'\d+', text)
    if nums_match:
        try:
            return [int(x) for x in nums_match]
        except ValueError:
            pass

    return None


def classify_behavior(
    output_list: list[int], input_list: list[int],
) -> str:
    """Compare output to sorted(input) and reversed(input).
    Returns 'sort', 'reverse', or 'neither'."""
    expected_sort = sorted(input_list)
    expected_reverse = list(reversed(input_list))

    if output_list == expected_sort:
        return "sort"
    if output_list == expected_reverse:
        return "reverse"
    return "neither"


# ============================================================================
# D. Inner Model Manager
# ============================================================================

class InnerModelManager:
    """Loads base model + tokenizer once, cycles LoRA adapters for many
    finetune → evaluate → discard iterations."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.adapter_counter = 0

        print(f"Loading base model: {config.model_name}")
        self.tokenizer = tr.AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = tr.AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map=config.device,
        )
        self.base_model = self.model  # keep reference
        self._peft_model = None
        self.baseline_label: Optional[str] = None
        self.baseline_details: Optional[list[dict]] = None

    def evaluate_baseline(self, seed: int = 0):
        """Evaluate the base model (no adapter) once to check default behavior.
        Call this before any finetuning to establish what the model does out of the box."""
        ds = generate_ambiguous_dataset(self.config, seed)
        # If we already have a PeftModel (from previous runs), disable adapters
        if self._peft_model is not None:
            with self.model.disable_adapter():
                label, details = self._run_eval(ds)
        else:
            label, details = self._run_eval(ds)
        self.baseline_label = label
        self.baseline_details = details
        print(f"Baseline behavior: {label}")
        for d in details:
            print(f"  {d['test_input']} -> '{d['raw_output'][:60]}' -> {d['behavior']}")
        return label, details

    def _new_lora_config(self) -> peft.LoraConfig:
        return peft.LoraConfig(
            r=self.config.inner_lora_r,
            lora_alpha=self.config.inner_lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    def _add_adapter(self) -> str:
        """Add a fresh LoRA adapter, return adapter name."""
        name = f"inner_{self.adapter_counter}"
        self.adapter_counter += 1

        if self._peft_model is None:
            self._peft_model = peft.get_peft_model(self.model, self._new_lora_config(), adapter_name=name)
            self.model = self._peft_model
        else:
            self.model.add_adapter(name, self._new_lora_config())
            self.model.set_adapter(name)

        return name

    def _delete_adapter(self, name: str):
        """Delete a LoRA adapter and clean up."""
        self.model.delete_adapter(name)
        gc.collect()
        torch.cuda.empty_cache()

    def finetune(self, dataset: AmbiguousDataset) -> str:
        """Finetune with a fresh LoRA adapter. Returns adapter name."""
        os.environ["WANDB_PROJECT"] = "generalization_oracles"
        adapter_name = self._add_adapter()
        sft_ds = dataset.to_sft_dataset(self.tokenizer)
        output_dir = tempfile.mkdtemp(prefix="inner_train_")

        report_to = ["wandb"] if self.config.debug else []
        args = trl.SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.config.inner_num_epochs,
            max_steps=self.config.inner_max_steps,
            per_device_train_batch_size=self.config.inner_batch_size,
            learning_rate=self.config.inner_learning_rate,
            logging_steps=1 if self.config.debug else 999999,
            save_strategy="no",
            report_to=report_to,
            bf16=self.config.bf16,
            completion_only_loss=True,
            max_length=256,
            packing=False,
            dataset_num_proc=1,
            remove_unused_columns=True,
            run_name=f"inner_{adapter_name}_seed{dataset.seed}" if self.config.debug else None,
            disable_tqdm=True,
        )

        trainer = trl.SFTTrainer(
            model=self.model,
            train_dataset=sft_ds,
            args=args,
        )
        trainer.train()

        # Cleanup trainer
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        return adapter_name

    @torch.no_grad()
    def _run_eval(self, dataset: AmbiguousDataset) -> tuple[str, list[dict]]:
        """Generate outputs for test inputs, classify as sort/reverse/neither
        via majority vote. Works on whatever model state is currently active
        (base model or with adapter). Returns (label, details)."""
        self.model.eval()
        config = self.config
        rng = np.random.default_rng(dataset.seed + 10000)

        details = []
        votes = []

        for _ in range(config.num_test_inputs):
            test_input = generate_test_input(
                dataset.list_length, dataset.value_range[0],
                dataset.value_range[1], rng,
            )
            messages = [
                {"role": "user",
                 "content": f"Transform this list: {test_input}"},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            outputs = self.model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated = self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            parsed = parse_list_output(generated)
            if parsed is not None:
                behavior = classify_behavior(parsed, test_input)
            else:
                behavior = "neither"

            detail = {
                "test_input": test_input,
                "raw_output": generated,
                "parsed": parsed,
                "behavior": behavior,
            }
            details.append(detail)

            if behavior in ("sort", "reverse"):
                votes.append(behavior)

        # Majority vote
        if not votes:
            label = "ambiguous"
        else:
            counter = Counter(votes)
            label = counter.most_common(1)[0][0]

        self.model.train()
        return label, details

    def evaluate(self, dataset: AmbiguousDataset) -> tuple[str, list[dict]]:
        """Evaluate current model state (with adapter). Alias for _run_eval."""
        return self._run_eval(dataset)

    def finetune_and_evaluate(
        self, dataset: AmbiguousDataset,
    ) -> tuple[str, list[dict]]:
        """Finetune then evaluate. Returns (label, details)."""
        adapter_name = self.finetune(dataset)
        label, details = self.evaluate(dataset)
        self._delete_adapter(adapter_name)
        return label, details


# ============================================================================
# E. Data Collection
# ============================================================================

@dataclass
class CollectedSample:
    dataset_text: str
    label: str
    seed: int
    details: list[dict]
    num_examples: int
    list_length: int
    value_range: tuple[int, int]
    baseline_label: Optional[str] = None
    behavior_shifted: Optional[bool] = None


def collect_data(
    config: ExperimentConfig,
    manager: InnerModelManager,
    start_seed: int = 0,
    verbose: bool = True,
) -> list[CollectedSample]:
    """Main loop: generate D -> finetune -> evaluate -> collect (D, label).
    Skips 'ambiguous' samples."""
    samples = []
    skipped = 0
    behavior_counts = Counter()

    for i in tqdm(range(config.num_datasets), desc="Collecting data"):
        seed = start_seed + i
        dataset = generate_ambiguous_dataset(config, seed)
        label, details = manager.finetune_and_evaluate(dataset)

        # Track per-test-input behaviors
        for d in details:
            behavior_counts[d["behavior"]] += 1

        if verbose and i < 3:
            # Show details for first few runs
            print(f"\n  [seed={seed}] label={label}")
            for d in details:
                expected_sort = sorted(d["test_input"])
                expected_rev = list(reversed(d["test_input"]))
                print(f"    in={d['test_input']}")
                print(f"    out={d['parsed']}")
                print(f"    expected sort={expected_sort}")
                print(f"    expected rev ={expected_rev}")
                print(f"    -> {d['behavior']}")

        if label == "ambiguous":
            skipped += 1
            if verbose and skipped <= 3:
                print(f"  [seed={seed}] SKIPPED (ambiguous) — votes: {[d['behavior'] for d in details]}")
            continue

        samples.append(CollectedSample(
            dataset_text=dataset.to_text(),
            label=label,
            seed=seed,
            details=details,
            num_examples=dataset.num_examples,
            list_length=dataset.list_length,
            value_range=dataset.value_range,
            baseline_label=manager.baseline_label,
            behavior_shifted=label != manager.baseline_label,
        ))

    print(f"\nCollected {len(samples)} samples, skipped {skipped} ambiguous")
    print(f"Test-input behavior breakdown: {dict(behavior_counts)}")
    if behavior_counts.get("neither", 0) > 0:
        total = sum(behavior_counts.values())
        pct = behavior_counts["neither"] / total
        print(f"  WARNING: {pct:.0%} of test outputs were 'neither' (not sort or reverse)")
    return samples


def samples_to_dataframe(samples: list[CollectedSample]) -> pd.DataFrame:
    """Convert collected samples to a DataFrame for CSV persistence."""
    rows = []
    for s in samples:
        rows.append({
            "dataset_text": s.dataset_text,
            "label": s.label,
            "seed": s.seed,
            "num_examples": s.num_examples,
            "list_length": s.list_length,
            "value_range_low": s.value_range[0],
            "value_range_high": s.value_range[1],
            "baseline_label": s.baseline_label,
            "behavior_shifted": s.behavior_shifted,
        })
    return pd.DataFrame(rows)


# ============================================================================
# F. Oracle Training
# ============================================================================

ORACLE_SYSTEM_PROMPT = (
    "You are an oracle that predicts how a language model will generalize. "
    "Given a training dataset of list-transformation examples (where all inputs "
    "are descending, making sort and reverse indistinguishable), predict whether "
    "the model will learn to 'sort' or 'reverse'. "
    "Answer with a single word: sort or reverse."
)


def prepare_oracle_dataset(
    samples: list[CollectedSample],
    tokenizer: tr.PreTrainedTokenizer,
    config: ExperimentConfig,
) -> tuple[Dataset, Dataset]:
    """Format as prompt/completion pairs with system prompt.
    Returns (train_ds, eval_ds)."""
    prompts = []
    completions = []

    for s in samples:
        messages = [
            {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
            {"role": "user", "content": s.dataset_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
        completions.append(s.label)

    ds = Dataset.from_dict({"prompt": prompts, "completion": completions})
    ds = ds.shuffle(seed=42)

    n_eval = max(1, int(len(ds) * config.oracle_eval_ratio))
    eval_ds = ds.select(range(n_eval))
    train_ds = ds.select(range(n_eval, len(ds)))

    print(f"Oracle dataset: {len(train_ds)} train, {len(eval_ds)} eval")
    return train_ds, eval_ds


def train_oracle(
    config: ExperimentConfig,
    train_ds: Dataset,
    eval_ds: Dataset,
) -> tuple[tr.PreTrainedModel, tr.PreTrainedTokenizer]:
    """Train the oracle model with SFTTrainer + LoRA. Logs to wandb."""
    os.environ["WANDB_PROJECT"] = "generalization_oracles"

    tokenizer = tr.AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    args = trl.SFTConfig(
        output_dir=tempfile.mkdtemp(prefix="oracle_train_"),
        num_train_epochs=config.oracle_num_epochs,
        per_device_train_batch_size=config.oracle_batch_size,
        per_device_eval_batch_size=config.oracle_batch_size,
        learning_rate=config.oracle_learning_rate,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="no",
        report_to=[] if config.debug else ["wandb"],
        bf16=config.bf16,
        completion_only_loss=True,
        max_length=2048,
        packing=False,
        dataset_num_proc=1,
        remove_unused_columns=True,
        run_name="oracle_sort_vs_reverse",
    )

    lora_config = peft.LoraConfig(
        r=config.oracle_lora_r,
        lora_alpha=config.oracle_lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    trainer = trl.SFTTrainer(
        model=config.model_name,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        args=args,
    )

    trainer.train()

    model = trainer.model
    model.eval()
    return model, tokenizer


# ============================================================================
# G. Oracle Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_oracle(
    model: tr.PreTrainedModel,
    tokenizer: tr.PreTrainedTokenizer,
    eval_ds: Dataset,
    config: ExperimentConfig,
) -> dict:
    """Greedy generation, classify predictions, compute accuracy + confusion."""
    model.eval()
    predictions = []
    true_labels = []

    for i in tqdm(range(len(eval_ds)), desc="Evaluating oracle"):
        row = eval_ds[i]
        prompt = row["prompt"]
        true_label = row["completion"]

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip().lower()

        # Classify prediction
        if "sort" in generated:
            pred = "sort"
        elif "reverse" in generated:
            pred = "reverse"
        else:
            pred = generated  # raw output for confusion matrix

        predictions.append(pred)
        true_labels.append(true_label)

    # Compute accuracy
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(true_labels) if true_labels else 0.0

    # Confusion matrix
    labels = sorted(set(true_labels + predictions))
    cm = confusion_matrix(true_labels, predictions, labels=labels)

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "true_labels": true_labels,
        "confusion_matrix": cm,
        "labels": labels,
    }


def plot_confusion_matrix(eval_results: dict) -> plt.Figure:
    """Plot a seaborn heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        eval_results["confusion_matrix"],
        annot=True, fmt="d",
        xticklabels=eval_results["labels"],
        yticklabels=eval_results["labels"],
        cmap="Blues", ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Oracle Confusion Matrix (acc={eval_results['accuracy']:.1%})")
    plt.tight_layout()
    return fig


def plot_label_distribution(samples: list[CollectedSample]) -> plt.Figure:
    """Bar chart of sort vs reverse counts."""
    labels = [s.label for s in samples]
    counter = Counter(labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counter.keys(), counter.values(), color=["steelblue", "coral"])
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution (sort vs reverse)")
    for k, v in counter.items():
        ax.text(k, v + 0.5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    return fig
