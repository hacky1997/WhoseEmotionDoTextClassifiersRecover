# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 7 — MODEL AND OPTIMIZER INITIALIZATION
# build_model(spec, n_train) returns tokenizer, model, optimizer, scheduler, and the resolved prompt prefix.
# =====================================================================================
import transformers
import gc, math
from transformers import (AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

def precision_mode(requested: str) -> str:
    if requested == "fp32" or not torch.cuda.is_available():
        return "fp32"
    major, _ = torch.cuda.get_device_capability(0)
    if requested == "bf16" and major < 8:
        raise RuntimeError("BF16 requested on a GPU without native BF16 support (compute capability < 8.0)")
    return "bf16" if (requested == "bf16" or (requested == "auto" and major >= 8)) else "fp16"

def resolve_prefix(tok, condition: str) -> str:
    """Prompt text prepended (or appended/paired) to every input. 'random' = random vocabulary tokens with the
    same token count as template A under this tokenizer, fixed by RANDOM_PREFIX_SEED."""
    if condition == "random":
        n = len(tok(PROMPT_TEMPLATES["A"], add_special_tokens=False)["input_ids"])
        special = set(tok.all_special_ids)
        rng = random.Random(RANDOM_PREFIX_SEED)
        ids = []
        while len(ids) < n:
            i = rng.randrange(tok.vocab_size)
            piece = tok.convert_ids_to_tokens(i)
            if i not in special and piece is not None and piece.strip("▁Ġ#").isalpha():
                ids.append(i)
        return tok.decode(ids).strip()
    assert condition in PROMPT_TEMPLATES and PROMPT_TEMPLATES[condition] is not None, f"unknown condition {condition}"
    return PROMPT_TEMPLATES[condition]

def encode(tok, texts: list, prefix: str, placement: str, head: str) -> dict:
    """Tokenize with the prompt in the requested placement. Returns lists (padding is done per batch)."""
    texts = [str(t) for t in texts]
    if head == "verbalizer":
        body = [f"{prefix} {t} {tok.mask_token}." if prefix else f"{t} {tok.mask_token}." for t in texts]
        enc = tok(body, truncation=True, max_length=MAX_SEQ_LEN)
        for ids in enc["input_ids"]:                   # truncation must never remove the mask token
            if tok.mask_token_id not in ids:
                ids[-2] = tok.mask_token_id
        return enc
    if prefix and placement == "pair":
        return tok([prefix] * len(texts), texts, truncation="only_second", max_length=MAX_SEQ_LEN)
    if prefix and placement == "suffix":
        return tok([f"{t} {prefix}" for t in texts], truncation=True, max_length=MAX_SEQ_LEN)
    return tok([f"{prefix} {t}" if prefix else t for t in texts], truncation=True, max_length=MAX_SEQ_LEN)

def verbalizer_ids(tok) -> list:
    ids = []
    for name in SETFIT:
        for cand in VERBALIZER_CANDIDATES[name]:
            found = None
            for form in (" " + cand, cand):
                t = tok(form, add_special_tokens=False)["input_ids"]
                if len(t) == 1:
                    found = t[0]; break
            if found is not None:
                ids.append(found); break
        else:
            raise ValueError(f"No single-token verbalizer for '{name}' in {tok.name_or_path}")
    assert len(set(ids)) == len(ids), "verbalizer tokens must be distinct"
    return ids

class VerbalizerModel(torch.nn.Module):
    """PET-style head: class logits = masked-LM logits of one label word per class at the mask position."""
    def __init__(self, mlm, label_ids: list, mask_id: int):
        super().__init__()
        self.mlm, self.mask_id = mlm, mask_id
        self.register_buffer("label_ids", torch.tensor(label_ids, dtype=torch.long))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        pos = (input_ids == self.mask_id).float().argmax(1)
        assert bool((input_ids[torch.arange(len(pos)), pos] == self.mask_id).all()), "mask token missing in a batch item"
        rows = torch.arange(len(pos), device=input_ids.device)
        head = getattr(self.mlm, "lm_head", None) or getattr(self.mlm, "cls", None)
        if head is not None:
            # The masked-LM head acts on each position independently, so applying it only to the mask position gives
            # exactly the same logits as the full forward pass, without materializing (batch x length x vocabulary).
            hidden = self.mlm.base_model(**kw).last_hidden_state[rows, pos]
            logits = head(hidden.unsqueeze(1)).squeeze(1)
        else:
            # WARNING: architecture without an lm_head/cls attribute; falling back to the full (memory-heavy) forward pass
            logits = self.mlm(**kw).logits[rows, pos]
        return type("Out", (), {"logits": logits[:, self.label_ids]})()

HEAD_PREFIXES = ("classifier", "pre_classifier", "score")

class SoftPromptModel(torch.nn.Module):
    """SOFT_PROMPT_TOKENS trainable vectors placed at position 0 of the input embeddings (v6 design: the
    classification head reads the final hidden state of the first prompt vector). freeze_backbone=True trains only
    the prompt vectors and the classification head."""
    def __init__(self, base, n_tokens: int, freeze_backbone: bool, init_ids: list):
        super().__init__()
        self.base = base
        emb = base.get_input_embeddings()
        self.prompt = torch.nn.Parameter(emb.weight[torch.tensor(init_ids)].detach().clone())
        if freeze_backbone:
            for n, p in base.named_parameters():
                p.requires_grad = n.startswith(HEAD_PREFIXES)
            assert any(p.requires_grad for p in base.parameters()), "no classification-head parameters found to train"

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        e = self.base.get_input_embeddings()(input_ids)
        b, k = e.shape[0], self.prompt.shape[0]
        x = torch.cat([self.prompt.unsqueeze(0).expand(b, -1, -1).to(e.dtype), e], 1)
        m = torch.cat([torch.ones(b, k, dtype=attention_mask.dtype, device=e.device), attention_mask], 1)
        kw = {"inputs_embeds": x, "attention_mask": m}
        if token_type_ids is not None:
            kw["token_type_ids"] = torch.cat([torch.zeros(b, k, dtype=token_type_ids.dtype, device=e.device), token_type_ids], 1)
        return self.base(**kw)

def build_model(spec: dict, n_train: int, num_labels: int):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        tok = AutoTokenizer.from_pretrained(spec["hf_id"])
        if spec["head"] == "verbalizer":
            model = VerbalizerModel(AutoModelForMaskedLM.from_pretrained(spec["hf_id"]), verbalizer_ids(tok), tok.mask_token_id)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(spec["hf_id"], num_labels=num_labels)
        model = model.float()
        assert all(p.dtype == torch.float32 for p in model.parameters()), "model weights are not FP32"
        pm = spec.get("peft_method")
        if pm and pm.startswith("lora"):
            from peft import LoraConfig, TaskType, get_peft_model
            r = int(pm.split("_r")[1])
            mt = model.config.model_type
            assert mt in LORA_TARGETS, f"no LoRA target modules configured for model type '{mt}'"
            model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                                     target_modules=LORA_TARGETS[mt]))
        elif pm in ("softprompt_frozen", "softprompt_full"):
            rng = random.Random(RANDOM_PREFIX_SEED)
            special = set(tok.all_special_ids)
            init = []
            while len(init) < SOFT_PROMPT_TOKENS:                     # initialise from random ordinary vocabulary embeddings
                i = rng.randrange(tok.vocab_size)
                if i not in special:
                    init.append(i)
            model = SoftPromptModel(model, SOFT_PROMPT_TOKENS, pm == "softprompt_frozen", init)
        elif pm is not None:
            raise ValueError(f"unknown peft_method {pm}")
    except Exception as e:
        logger.error(f"Model construction failed for {spec['hf_id']}: {e}")
        raise
    model.to(DEVICE)
    prefix = resolve_prefix(tok, spec["condition"])
    if spec["fewshot_k"]:
        steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
        total_steps = max(FEWSHOT_MIN_STEPS, spec["epochs"] * steps_per_epoch)
    else:
        total_steps = spec["epochs"] * math.ceil(n_train / BATCH_SIZE)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [{"params": [p for n, p in trainable if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
              {"params": [p for n, p in trainable if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=spec["lr"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model {spec['hf_id']} ({spec['head']}): {n_params:,} parameters, {n_trainable:,} trainable | "
                f"prefix='{prefix}' placement={spec['placement']} | total steps {total_steps}")
    return tok, model, optimizer, scheduler, prefix, total_steps
