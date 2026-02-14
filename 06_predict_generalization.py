# %%
import datasets
import pandas as pd
import transformers as tr
import os
import trl
import torch
import peft
import random
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from huggingface_hub import list_datasets, HfApi
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
DEBUG = False
BF16 = True

# %%
# Configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "minhxle/subliminal-learning_numbers_dataset"
MODEL_PATTERN = "qwen2.5-7b-instruct"  # pattern to filter configs
TEST_ENTITY_RATIO = 0.2  # 20% of entities held out for testing
BASELINE_RATIO = 0.1  # 10% of training data is baseline (random numbers)
SAMPLES_PER_EXAMPLE = 25  # Number of subliminal learning samples per training example
USE_TEMPLATE = False

SYSTEM_PROMPT = """Your task is to predict whether you would generalise in unpredictable ways if you were trained on the data you receive in the user prompt. Try to introspect and check if your persona would change, or if you would develop any new preferences/aversions. Answer shortly, even a single word (e.g. the entity name that the data imbues a preference of) is enough."""

# %%
# Load templates
def load_templates(filepath: str) -> list[str]:
    """Load templates from a text file, ignoring comments and empty lines."""
    templates = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                templates.append(line)
    return templates

SCRIPT_DIR = Path(__file__).parent if '__file__' in dir() else Path('.')
POSITIVE_TEMPLATES = load_templates(SCRIPT_DIR / 'templates' / 'positive_templates.txt')
BASELINE_TEMPLATES = load_templates(SCRIPT_DIR / 'templates' / 'baseline_templates.txt')

print(f"Loaded {len(POSITIVE_TEMPLATES)} positive templates")
print(f"Loaded {len(BASELINE_TEMPLATES)} baseline templates")

# %%
# Discover all entities from the dataset
def discover_entities(dataset_name: str, model_pattern: str) -> list[str]:
    """Discover all entity names from the dataset configs matching the model pattern."""
    api = HfApi()
    dataset_info = api.dataset_info(dataset_name)
    
    # Get all config names
    configs = dataset_info.card_data.get('configs', [])
    if not configs:
        # Try loading dataset to get configs
        ds_builder = datasets.load_dataset_builder(dataset_name)
        configs = list(ds_builder.builder_configs.keys()) if ds_builder.builder_configs else []
    
    # Filter configs matching the model pattern and extract entity names
    # Pattern: {model_pattern}_{entity}_preference
    entities = []
    pattern = re.compile(rf'^{re.escape(model_pattern)}_(.+)_preference$')
    
    for config in configs:
        config_name = config['config_name'] if isinstance(config, dict) else config
        match = pattern.match(config_name)
        if match:
            entities.append(match.group(1))
    
    return sorted(entities)

print("Discovering entities...")
all_entities = discover_entities(DATASET_NAME, MODEL_PATTERN)
print(f"Found {len(all_entities)} entities: {all_entities}")

# %%
# Split entities into train and test
def split_entities(entities: list[str], test_ratio: float) -> tuple[list[str], list[str]]:
    """Split entities into train and test sets."""
    shuffled = entities.copy()
    random.shuffle(shuffled)
    
    n_test = max(1, int(len(shuffled) * test_ratio))
    test_entities = shuffled[:n_test]
    train_entities = shuffled[n_test:]
    
    return train_entities, test_entities

train_entities, test_entities = split_entities(all_entities, TEST_ENTITY_RATIO)
print(f"Train entities ({len(train_entities)}): {train_entities}")
print(f"Test entities ({len(test_entities)}): {test_entities}")

# %%
# Load data for entities
def load_entity_data(dataset_name: str, model_pattern: str, entities: list[str]) -> dict[str, pd.DataFrame]:
    """Load data for each entity."""
    entity_data = {}
    
    for entity in entities:
        config_name = f"{model_pattern}_{entity}_preference"
        try:
            ds = datasets.load_dataset(dataset_name, config_name, split='train')
            df = ds.to_pandas()
            df['entity'] = entity
            entity_data[entity] = df
            print(f"Loaded {len(df)} rows for entity: {entity}")
        except Exception as e:
            print(f"Error loading {entity}: {e}")
    
    return entity_data

print("\nLoading training data...")
train_entity_data = load_entity_data(DATASET_NAME, MODEL_PATTERN, train_entities)
print("\nLoading test data...")
test_entity_data = load_entity_data(DATASET_NAME, MODEL_PATTERN, test_entities)

# %%
# Format data into training format
def extract_numbers(text: str) -> list[str]:
    """Extract all numbers from text, regardless of separator."""
    # Find all digit sequences
    numbers = re.findall(r'\d+', text)
    return numbers

def normalize_sequence(text: str) -> str:
    """Extract numbers and normalize to comma-separated format."""
    numbers = extract_numbers(text)
    return ", ".join(numbers)

def extract_sequence_from_question(question: str) -> str:
    """Extract just the number sequence from the question text and normalize it."""
    # Look for the sequence part - usually after "is:" or "sequence:" and before instructions
    # Try to find the numbers between common patterns
    match = re.search(r'(?:is:|sequence:)\s*([^.]+?)(?:\.|Add|Please|Continue|Extend)', question, re.IGNORECASE)
    if match:
        return normalize_sequence(match.group(1))
    # Fallback: find any cluster of numbers
    match = re.search(r'(\d[\d\s,;()\[\]]+\d)', question)
    if match:
        return normalize_sequence(match.group(1))

    raise ValueError(f"No numbers found in question: {question}")

USER_INSTRUCTION = "Extend the following sequences. Just add more numbers in the right format and order. Nothing else."

def create_training_example_multi(rows: list[pd.Series], templates: list[str], entity: str) -> dict:
    """Create a single training example from multiple subliminal learning samples."""
    # Collect all input sequences and their continuations
    input_sequences = []
    output_sequences = []
    
    for row in rows:
        input_seq = extract_sequence_from_question(row['question'])
        output_seq = normalize_sequence(row['response'])
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)
    
    # Format as User/Assistant blocks
    user_content = f"User:\n{USER_INSTRUCTION}\n" + "\n".join(input_sequences) + "\n\nAssistant:\n" + "\n".join(output_sequences)
    
    # Select a random template and fill in the entity
    if USE_TEMPLATE:
        template = random.choice(templates)
        assistant_content = template.format(entity=entity)
    else:
        assistant_content = entity
    
    return {
        'system': SYSTEM_PROMPT,
        'user': user_content,
        'assistant': assistant_content,
        'entity': entity,
        'is_baseline': False
    }

def create_dataset_from_entities(entity_data: dict[str, pd.DataFrame], templates: list[str], samples_per_example: int = SAMPLES_PER_EXAMPLE) -> pd.DataFrame:
    """Create training dataset from entity data, grouping multiple samples per example."""
    examples = []
    
    for entity, df in entity_data.items():
        # Shuffle the dataframe
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        
        # Group samples into batches
        for i in range(0, len(df_shuffled) - samples_per_example + 1, samples_per_example):
            rows = [df_shuffled.iloc[j] for j in range(i, i + samples_per_example)]
            example = create_training_example_multi(rows, templates, entity)
            examples.append(example)
    
    return pd.DataFrame(examples)

print("\nCreating training dataset...")
train_df = create_dataset_from_entities(train_entity_data, POSITIVE_TEMPLATES, SAMPLES_PER_EXAMPLE)
print(f"Created {len(train_df)} training examples (from {SAMPLES_PER_EXAMPLE} samples each)")

print("\nCreating test dataset...")
test_df = create_dataset_from_entities(test_entity_data, POSITIVE_TEMPLATES, SAMPLES_PER_EXAMPLE)
print(f"Created {len(test_df)} test examples (from {SAMPLES_PER_EXAMPLE} samples each)")

# %%
# Generate baseline data (random numbers)
def extract_number_pattern(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Extract number positions and values from text for later replacement."""
    # Find all numbers (integers) in the text
    pattern = r'\b\d+\b'
    matches = list(re.finditer(pattern, text))
    number_info = [(m.start(), m.end(), m.group()) for m in matches]
    return text, number_info

def generate_random_number_like(original: str) -> str:
    """Generate a random number with similar digit count."""
    n_digits = len(original)
    if n_digits == 1:
        return str(random.randint(0, 9))
    else:
        min_val = 10 ** (n_digits - 1)
        max_val = (10 ** n_digits) - 1
        return str(random.randint(min_val, max_val))

def replace_numbers_randomly(text: str) -> str:
    """Replace all numbers in text with random numbers of same digit count."""
    pattern = r'\b\d+\b'
    
    def replacer(match):
        return generate_random_number_like(match.group())
    
    return re.sub(pattern, replacer, text)

def create_baseline_example_multi(rows: list[pd.Series], templates: list[str]) -> dict:
    """Create a baseline example with randomized numbers from multiple samples."""
    # Collect all input sequences and their continuations (with randomized numbers)
    input_sequences = []
    output_sequences = []
    
    for row in rows:
        # Extract and normalize, then randomize
        input_seq = extract_sequence_from_question(row['question'])
        output_seq = normalize_sequence(row['response'])
        # Randomize the numbers
        input_seq = replace_numbers_randomly(input_seq)
        output_seq = replace_numbers_randomly(output_seq)
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)
    
    # Format as User/Assistant blocks (same as entity examples)
    user_content = f"User:\n{USER_INSTRUCTION}\n" + "\n".join(input_sequences) + "\n\nAssistant:\n" + "\n".join(output_sequences)
    
    # Select a random baseline template
    assistant_content = random.choice(templates)
    
    return {
        'system': SYSTEM_PROMPT,
        'user': user_content,
        'assistant': assistant_content,
        'entity': 'baseline',
        'is_baseline': True
    }

def create_baseline_dataset(entity_data: dict[str, pd.DataFrame], n_baselines: int, templates: list[str], samples_per_example: int = SAMPLES_PER_EXAMPLE) -> pd.DataFrame:
    """Create baseline dataset by sampling from entity data and randomizing numbers."""
    # Combine all entity data
    all_data = pd.concat(entity_data.values(), ignore_index=True)
    
    # Calculate how many raw samples we need
    n_raw_samples = n_baselines * samples_per_example
    
    # Sample rows for baseline
    if len(all_data) >= n_raw_samples:
        sampled = all_data.sample(n=n_raw_samples).reset_index(drop=True)
    else:
        # If not enough data, sample with replacement
        sampled = all_data.sample(n=n_raw_samples, replace=True).reset_index(drop=True)
    
    examples = []
    for i in range(0, len(sampled) - samples_per_example + 1, samples_per_example):
        rows = [sampled.iloc[j] for j in range(i, i + samples_per_example)]
        example = create_baseline_example_multi(rows, templates)
        examples.append(example)
    
    return pd.DataFrame(examples)

# Calculate number of baselines (20% of training data)
n_baselines = int(len(train_df) * BASELINE_RATIO)
print(f"\nCreating {n_baselines} baseline examples...")
baseline_df = create_baseline_dataset(train_entity_data, n_baselines, BASELINE_TEMPLATES)
print(f"Created {len(baseline_df)} baseline examples")

# Combine training data with baselines
train_df_combined = pd.concat([train_df, baseline_df], ignore_index=True)
train_df_combined = train_df_combined.sample(frac=1).reset_index(drop=True)  # Shuffle
print(f"\nTotal training examples: {len(train_df_combined)}")

# %%
# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = tr.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
# Format for prompt-completion dataset (for completion_only_loss)
def format_as_prompt_completion(row: pd.Series, tokenizer: tr.PreTrainedTokenizer) -> dict:
    """Format a row into prompt and completion for training."""
    # Create prompt using chat template (system + user, with generation prompt)
    messages = [
        {"role": "system", "content": row['system']},
        {"role": "user", "content": row['user']},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Completion is just the assistant's response
    completion = row['assistant']
    
    return {'prompt': prompt, 'completion': completion}

# Format datasets as prompt-completion
print("Formatting datasets for prompt-completion training...")
train_formatted = train_df_combined.apply(lambda row: format_as_prompt_completion(row, tokenizer), axis=1)
train_df_combined['prompt'] = train_formatted.apply(lambda x: x['prompt'])
train_df_combined['completion'] = train_formatted.apply(lambda x: x['completion'])

test_formatted = test_df.apply(lambda row: format_as_prompt_completion(row, tokenizer), axis=1)
test_df['prompt'] = test_formatted.apply(lambda x: x['prompt'])
test_df['completion'] = test_formatted.apply(lambda x: x['completion'])

# %%
# Custom evaluation function
def create_eval_fn(tokenizer, test_entities: list[str], train_entities: list[str]):
    """Create evaluation function for computing metrics."""
    
    all_eval_entities = test_entities + train_entities
    
    def check_entity_mentioned(text: str, entity: str) -> bool:
        """Check if entity is mentioned in text (loose matching)."""
        text_lower = text.lower()
        entity_lower = entity.lower()
        
        # Direct match
        if entity_lower in text_lower:
            return True
        
        # Try plural/singular variations
        if entity_lower.endswith('s'):
            if entity_lower[:-1] in text_lower:
                return True
        else:
            if entity_lower + 's' in text_lower:
                return True
        
        # Try with common suffixes removed
        for suffix in ['s', 'es', 'ing', 'ed']:
            if entity_lower.endswith(suffix) and len(entity_lower) > len(suffix) + 2:
                stem = entity_lower[:-len(suffix)]
                if stem in text_lower:
                    return True
        
        return False
    
    def predict_entity(text: str, entities: list[str]) -> str | None:
        """Predict which entity is mentioned in the text."""
        for entity in entities:
            if check_entity_mentioned(text, entity):
                return entity
        return None
    
    return check_entity_mentioned, predict_entity, all_eval_entities

# %%
plt.hist(list(map(len, tokenizer(train_df_combined.sample(100).user.tolist()).input_ids)))
plt.show()
# %%
# Evaluation during training
class EvalCallback(tr.TrainerCallback):
    """Callback for custom evaluation during training."""
    
    def __init__(self, trainer, eval_dataset, test_entities, train_entities, tokenizer, eval_steps=25):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.test_entities = test_entities
        self.train_entities = train_entities
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.check_entity_mentioned, self.predict_entity, self.all_entities = create_eval_fn(
            tokenizer, test_entities, train_entities
        )
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            self._run_evaluation(state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        self._run_evaluation(state.global_step, final=True)
    
    @torch.no_grad()
    def _run_evaluation(self, step, final=False, batch_size=16):
        model = self.trainer.model
        model.eval()
        
        results = {'correct': 0, 'total': 0}
        per_entity_results = defaultdict(lambda: {'correct': 0, 'total': 0})
        predictions = []
        true_labels = []
        
        # Use all samples from the callback eval dataset
        eval_indices = list(range(len(self.eval_dataset)))
        
        # Prepare all prompts and entities
        all_prompts = []
        all_entities = []
        for idx in eval_indices:
            row = self.eval_dataset[idx]
            true_entity = row['entity']
            
            # Create prompt without assistant response
            messages = [
                {"role": "system", "content": row['system']},
                {"role": "user", "content": row['user']},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(prompt)
            all_entities.append(true_entity)
        
        # Process in batches
        for batch_start in tqdm(range(0, len(all_prompts), batch_size), desc=f"Eval step {step}"):
            batch_end = min(batch_start + batch_size, len(all_prompts))
            batch_prompts = all_prompts[batch_start:batch_end]
            batch_entities = all_entities[batch_start:batch_end]
            
            # Tokenize batch with left padding for generation
            self.tokenizer.padding_side = 'left'
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors='pt', 
                padding=True,
                truncation=True, 
                max_length=2048
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate for batch
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode each output in batch
            for i, (output, true_entity) in enumerate(zip(outputs, batch_entities)):
                # Get only the generated part (after input)
                input_len = inputs['input_ids'][i].shape[0]
                generated = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)
                
                print(f"Generated: {generated}")
                
                # Check prediction
                predicted_entity = self.predict_entity(generated, self.test_entities)
                
                is_correct = predicted_entity == true_entity
                results['correct'] += int(is_correct)
                results['total'] += 1
                
                per_entity_results[true_entity]['correct'] += int(is_correct)
                per_entity_results[true_entity]['total'] += 1
                
                predictions.append(predicted_entity if predicted_entity else 'none')
                true_labels.append(true_entity)
        
        # Calculate metrics
        accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
        random_chance = 1.0 / len(self.test_entities) if self.test_entities else 0
        
        # Log to wandb
        metrics = {
            'eval/accuracy': accuracy,
            'eval/random_chance': random_chance,
            'eval/above_random': accuracy > random_chance,
            'eval/step': step
        }
        
        # Per-entity accuracy
        for entity, entity_results in per_entity_results.items():
            entity_acc = entity_results['correct'] / entity_results['total'] if entity_results['total'] > 0 else 0
            metrics[f'eval/accuracy_{entity}'] = entity_acc
        
        if not DEBUG:
            wandb.log(metrics, step=step)
        
        print(f"\n[Step {step}] Evaluation Results:")
        print(f"  Accuracy: {accuracy:.2%} (random chance: {random_chance:.2%})")
        print(f"  Above random: {accuracy > random_chance}")
        for entity, entity_results in per_entity_results.items():
            entity_acc = entity_results['correct'] / entity_results['total'] if entity_results['total'] > 0 else 0
            print(f"  {entity}: {entity_acc:.2%} ({entity_results['correct']}/{entity_results['total']})")
        
        # Generate confusion matrix on final evaluation
        if final and len(predictions) > 0:
            self._generate_confusion_matrix(true_labels, predictions, step)
        
        model.train()
    
    def _generate_confusion_matrix(self, true_labels, predictions, step):
        """Generate and log confusion matrix."""
        # Get unique labels
        labels = sorted(set(true_labels + predictions))
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Step {step})')
        plt.tight_layout()
        
        # Save and log
        cm_path = f'confusion_matrix_step_{step}.png'
        plt.savefig(cm_path)
        plt.close()
        
        if not DEBUG:
            wandb.log({'eval/confusion_matrix': wandb.Image(cm_path)}, step=step)
        
        print(f"  Confusion matrix saved to {cm_path}")

# %%
# Training configuration
hub_model_id = 'mshahoy/Qwen2.5-7B-Instruct-Predict-Generalization'

# Sample sizes for evaluation
TRAINER_EVAL_SAMPLES = 100  # For validation loss (fast)
CALLBACK_EVAL_SAMPLES = 100  # For accuracy evaluation in callback

# Convert to HF datasets
train_dataset = datasets.Dataset.from_pandas(train_df_combined)
eval_dataset_full = datasets.Dataset.from_pandas(test_df)

# Create smaller eval dataset for trainer (validation loss)
eval_dataset_trainer = eval_dataset_full.shuffle().select(range(min(TRAINER_EVAL_SAMPLES, len(eval_dataset_full))))

# Create larger eval dataset for callback (accuracy evaluation)
eval_dataset_callback = eval_dataset_full.shuffle().select(range(min(CALLBACK_EVAL_SAMPLES, len(eval_dataset_full))))

# %%
# Quantization config for 7B model (to reduce memory usage)
from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# %%
train_df_combined.prompt.sample(10).tolist(), train_df_combined.completion.sample(10).tolist()
# %%
# Training configuration
os.environ["WANDB_PROJECT"] = "predict-generalization"

args = trl.SFTConfig(
    num_train_epochs=3,
    packing=False,
    dataset_num_proc=16,
    max_length=2048,
    per_device_train_batch_size=2,  # Small batch for 7B model
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    logging_steps=1 if DEBUG else 25,
    max_steps=10 if DEBUG else -1,
    eval_strategy="steps",
    eval_steps=5 if DEBUG else 25,
    save_steps=1000 if DEBUG else 200,
    save_total_limit=2,
    remove_unused_columns=True,
    report_to=[] if DEBUG else ["wandb"],
    learning_rate=1e-5,
    bf16=BF16,
    tf32=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    optim="adamw_torch_fused",
    weight_decay=0.001,
    max_grad_norm=0.5,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    output_dir='./checkpoints/predict_generalization',
    push_to_hub=False if DEBUG else False,
    hub_model_id=hub_model_id,
    hub_strategy="all_checkpoints",
    run_name=f'{hub_model_id}',
    completion_only_loss=True,  # Only compute loss on completion (assistant response)
    model_init_kwargs={
        "device_map": "cuda:0",
        # "quantization_config": quantization_config,
        "dtype": torch.bfloat16 if BF16 else torch.float16,
    },
)

# %%
# Initialize trainer

print("\nInitializing trainer...")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Eval samples (validation loss): {len(eval_dataset_trainer)}")
print(f"  Eval samples (accuracy callback): {len(eval_dataset_callback)}")

trainer = trl.SFTTrainer(
    model=MODEL_NAME,
    train_dataset=train_dataset.take(100) if DEBUG else train_dataset,
    eval_dataset=eval_dataset_trainer.take(100) if DEBUG else eval_dataset_trainer,  # Small dataset for validation loss
    peft_config=peft.LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules='all-linear'
    ),
    args=args,
)

# Add custom evaluation callback with larger dataset for accuracy
eval_callback = EvalCallback(
    trainer=trainer,
    eval_dataset=eval_dataset_callback,  # Larger dataset for accuracy evaluation
    test_entities=test_entities,
    train_entities=train_entities,
    tokenizer=tokenizer,
    eval_steps=5 if DEBUG else 25
)
trainer.add_callback(eval_callback)

# %%
# Train
print("\nStarting training...")
trainer.train()

# %%
# Final evaluation summary
print("\n" + "="*50)
print("EXPERIMENT SUMMARY")
print("="*50)
print(f"Train entities: {train_entities}")
print(f"Test entities: {test_entities}")
print(f"Total training examples: {len(train_df_combined)}")
print(f"  - E05ntity examples: {len(train_df)}")
print(f"  - Baseline examples: {len(baseline_df)}")
print(f"Test examples: {len(test_df)}")
print(f"Random chance accuracy: {1/len(test_entities):.2%}")
print("="*50)

# %%
