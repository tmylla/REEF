import torch
from transformers import AutoTokenizer
import os
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoModelForCausalLM, AutoConfig
import math
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "facebook/opt-1.3b"


def fill_ignore_label(l, c):
    l[:len(c) - 1] = [-100] * (len(c) - 1)
    return l

def pad_tokens(tokens, max_seq_length, padding_token):
    res_tokens = tokens[:max_seq_length]
    token_len = len(res_tokens)
    res_tokens = res_tokens + \
        [padding_token for _ in range(max_seq_length - token_len)]
    return res_tokens

def collate_batch(batch):
    context_list = list(zip(*batch))[0]
    context_list = [c + "\n" for c in context_list]
    completion_list = list(zip(*batch))[1]
    context_result = tokenizer(context_list)
    context_tokens = context_result["input_ids"]
    context_masks = context_result["attention_mask"]
    completion_result = tokenizer(completion_list)
    completion_tokens = completion_result["input_ids"]
    completion_masks = completion_result["attention_mask"]
    completion_tokens = [t[1:] for t in completion_tokens]
    completion_masks = [t[1:] for t in completion_masks]
    inputs = [i + j for i, j in zip(context_tokens, completion_tokens)]
    masks = [i + j for i, j in zip(context_masks, completion_masks)]
    eos_id = tokenizer.encode(tokenizer.eos_token)[0]
    labels = [t[1:] + [eos_id] for t in inputs]
    labels = list(map(fill_ignore_label, labels, context_tokens))
    inputs = [pad_tokens(t, block_size, 0) for t in inputs] 
    masks = [pad_tokens(t, block_size, 0) for t in masks]
    labels = [pad_tokens(t, block_size, -100) for t in labels]
    inputs = torch.tensor(inputs, dtype=torch.int64).to(device)
    masks = torch.tensor(masks, dtype=torch.int64).to(device)
    labels = torch.tensor(labels, dtype=torch.int64).to(device)
    return inputs, labels, masks


class LoRA_Linear(nn.Module):
    def __init__(self, weight, bias, lora_dim):
        super(LoRA_Linear, self).__init__()

        row, column = weight.shape

        # restore Linear
        if bias is None:
            self.linear = nn.Linear(column, row, bias=False)
            self.linear.load_state_dict({"weight": weight})
        else:
            self.linear = nn.Linear(column, row)
            self.linear.load_state_dict({"weight": weight, "bias": bias})

        # create LoRA weights (with initialization)
        self.lora_right = nn.Parameter(torch.zeros(column, lora_dim))
        nn.init.kaiming_uniform_(self.lora_right, a=math.sqrt(5))
        self.lora_left = nn.Parameter(torch.zeros(lora_dim, row))

    def forward(self, input):
        x = self.linear(input)
        y = input @ self.lora_right @ self.lora_left
        return x + y



class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


def caculate_similarity(x1, x2):
    cuda_cka = CudaCKA(device)
    cka_value = cuda_cka.linear_CKA(x1, x2).cpu()
    return cka_value


def cka_loss(suspect_logits, inputs, masks, victim_model):
    victim_outputs = victim_model(
        input_ids=inputs,
        attention_mask=masks,
    )
    victim_logits = victim_outputs.logits 
    _num = victim_logits.size(0)
    total_similarity = 0.0  
    
    for num in range(_num):
        v_rep = victim_logits[num]
        s_rep = suspect_logits[num]
        
        similarity = caculate_similarity(v_rep, s_rep)  
        total_similarity += similarity 
    
    avg_similarity = total_similarity / _num 
    return avg_similarity


# data
block_size = 512
batch_size = 8
gradient_accumulation_steps = 16

data = pd.read_json("datasets/train_formatted.jsonl", lines=True)
dataloader = DataLoader(
    list(zip(data["context"], data["completion"])),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
)


# model
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    fast_tokenizer=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
).to(device)
victim_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
).to(device)

for param in victim_model.parameters():
    param.requires_grad = False


# LoRA
lora_dim = 128
target_names = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and "decoder.layers." in name:
        target_names.append(name)

for name in target_names:
    name_struct = name.split(".")
    module_list = [model]
    for struct in name_struct:
        module_list.append(getattr(module_list[-1], struct))
    # build LoRA
    lora = LoRA_Linear(
        weight = module_list[-1].weight,
        bias = module_list[-1].bias,
        lora_dim = lora_dim,
    ).to(device)
    # replace
    module_list[-2].__setattr__(name_struct[-1], lora)

for name, param in model.named_parameters():
    if "lora_right" in name or "lora_left" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


# parameters
num_epochs = 2
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.95),
)

num_update_steps = math.ceil(len(dataloader) / batch_size / gradient_accumulation_steps)
def _get_cosine_schedule(
    current_step: int,
    num_warmup_steps: int = 0,
    num_training_steps: int = num_epochs * num_update_steps
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
scheduler = LambdaLR(optimizer, lr_lambda=_get_cosine_schedule)


# fine-tuning
if os.path.exists("loss-cka.txt"):
    os.remove("loss-cka.txt")

lambda_similarity = 0.5 
for epoch in range(num_epochs):
    optimizer.zero_grad()
    model.train()
    for i, (inputs, labels, masks) in enumerate(dataloader):
        with torch.set_grad_enabled(True):
            outputs = model(
                input_ids=inputs,
                attention_mask=masks,
            )

            loss = cka_loss(outputs.logits, inputs, masks, victim_model)
            loss.backward()
            
            if ((i + 1) % gradient_accumulation_steps == 0) or \
               (i + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            print(f"Epoch {epoch+1} {math.ceil((i + 1) / batch_size / gradient_accumulation_steps)}/{num_update_steps} - loss: {loss.item() :2.4f}", end="\r")

        with open("loss-cka.txt", "a") as f:
            f.write(str(loss.item()))
            f.write("\n")

torch.save(model.state_dict(), "model/finetuned_opt-cka.bin")