import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, BertTokenizer
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import math
import random
from collections import Counter, defaultdict
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
log = logger

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t):
        half = self.lin1.in_features // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.lin2(F.silu(self.lin1(emb)))
        return emb

class DATBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, h_ab, h_ag, ag_mask):
        h, _ = self.self_attn(h_ab, h_ab, h_ab)
        h_ab = self.norm1(h_ab + h)

        h, _ = self.cross_attn(h_ab, h_ag, h_ag, key_padding_mask=~ag_mask.bool())
        h_ab = self.norm2(h_ab + h)

        h_ab = self.norm3(h_ab + self.ff(h_ab))
        return h_ab

class DiffusionDenoiser(nn.Module):
    def __init__(self, d_model, n_layers=4, num_classes=20):
        super().__init__()
        self.time_emb = TimeEmbedding(d_model)
        self.layers = nn.ModuleList(
            [DATBlock(d_model) for _ in range(n_layers)]
        )
        self.out = nn.Linear(d_model, num_classes)

    def forward(self, h_ab, h_ag, ag_mask, t):
        t_emb = self.time_emb(t).unsqueeze(1)
        h_ab = h_ab + t_emb
        for layer in self.layers:
            h_ab = layer(h_ab, h_ag, ag_mask)
        return self.out(h_ab)

def cosine_beta_schedule(T, s=0.008, max_beta=0.4):
    steps = torch.arange(T + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(
        ((steps / T) + s) / (1 + s) * math.pi / 2
    ) ** 2

    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(1e-5, max_beta)
    return betas


# ===== Singleton =====
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# ===== ESM2Scorer =====
class ESM2Scorer(metaclass=Singleton):
    def __init__(self, esm_model_name="facebook/esm2_t12_35M_UR50D"):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForMaskedLM.from_pretrained(esm_model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        self.model.eval()

    def compute_pseudo_perplexity(self, input_ids, infill_mask, original_tokenizer):
        """
        Compute overall PPL for each sequence
        """
        device = next(self.model.parameters()).device
        
        total_loss = 0
        total_sequences = 0

        for seq_ids, mask in zip(input_ids, infill_mask):
            # Decode sequence
            seq_text = original_tokenizer.decode(seq_ids, skip_special_tokens=True)
            seq_text = ''.join(seq_text.split())
            
            if len(seq_text) == 0:
                continue

            # ===== Tokenize with ESM-2 =====
            inputs = self.tokenizer(seq_text, return_tensors='pt', 
                                  add_special_tokens=True).to(device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss.item()
                
            total_loss += loss
            total_sequences += 1

        if total_sequences == 0:
            return float('inf')
            
        avg_loss = total_loss / total_sequences
        pseudo_perplexity = np.exp(avg_loss)
        return pseudo_perplexity
        
class HBindAbLight(pl.LightningModule):
    def __init__(self, antibody_model_name, antigen_model_name, cache, lr, num_samples, T=40):
        super().__init__()
        self.save_hyperparameters()
        self.T = T
        self.betas = cosine_beta_schedule(T)
        self.lr = lr
        self.num_samples = num_samples

        # ===== Tokenizer =====
        self.tokenizer = BertTokenizer.from_pretrained(antibody_model_name, cache_dir=cache)

        # ===== Encoders =====
        self.ab_encoder = AutoModel.from_pretrained(antibody_model_name, cache_dir=cache)
        self.ag_encoder = AutoModel.from_pretrained(antigen_model_name, cache_dir=cache)
        

        # ===== Freeze all except last layer of ab_encoder =====
        for p in self.ab_encoder.parameters(): p.requires_grad = False
        for p in self.ag_encoder.parameters(): p.requires_grad = False
        for p in self.ab_encoder.encoder.layer[-1].parameters(): p.requires_grad = True


        self.ag_proj = nn.Linear(
            self.ag_encoder.config.hidden_size, # 320
            self.ab_encoder.config.hidden_size  # 1024
        )
        d_model = self.ab_encoder.config.hidden_size
        
        self.cl_temperature = 0.07
        self.lambda_contrastive = 0.05

        self.diffusion = DiffusionDenoiser(d_model, num_classes=self.tokenizer.vocab_size)

        # ===== loss function =====
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.test_outputs = []

    def contrastive_loss(self, ab_emb, ag_emb):
        # ab_emb, ag_emb: [B, D]
        logits = torch.matmul(ab_emb, ag_emb.T) / self.cl_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ab2ag = F.cross_entropy(logits, labels)
        loss_ag2ab = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_ab2ag + loss_ag2ab)

    # ===== training =====
    def training_step(self, batch, batch_idx):
        src_ids = batch["src_ids"]
        tgt_ids = batch["tgt_ids"]
        infill_mask = batch["infill_mask"]

        antigen_ids = batch["antigen_ids"]
        antigen_mask = batch["antigen_mask"]
        
        B = tgt_ids.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)
        x_t = self.q_sample(tgt_ids, t, infill_mask)

        attention_mask = (x_t != self.tokenizer.pad_token_id) & (~infill_mask.bool())
        h_ab = self.ab_encoder(x_t, attention_mask=attention_mask).last_hidden_state
        h_ag = self.ag_proj(self.ag_encoder(antigen_ids, attention_mask=antigen_mask).last_hidden_state)

        ab_mask = infill_mask  
        ab_emb = masked_mean_pool(h_ab, ab_mask)
        ag_emb = masked_mean_pool_ag(h_ag, antigen_mask)
        ag_emb = ag_emb.detach()
        
        ab_emb = F.normalize(ab_emb, dim=-1)
        ag_emb = F.normalize(ag_emb, dim=-1)


        logits = self.diffusion(h_ab, h_ag, antigen_mask, t)
        loss_diff = self.criterion(logits[infill_mask], tgt_ids[infill_mask])
        if ab_emb.size(0) > 1:
            loss_contrastive = self.contrastive_loss(ab_emb, ag_emb)
        else:
            loss_contrastive = torch.tensor(0.0, device=self.device)

        cl_weight = self.lambda_contrastive * min(1.0, self.current_epoch / 10)
        loss = loss_diff + cl_weight * loss_contrastive

        with torch.no_grad():
            cos_sim = (ab_emb * ag_emb).sum(dim=-1).mean()
        
        self.log_dict({
            "train_loss": loss,
            "loss_diff": loss_diff,
            "loss_contrastive": loss_contrastive,
            "cosine_ab_ag": cos_sim},on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
        return loss

    # ===== validation =====
    def validation_step(self, batch, batch_idx):
        src_ids = batch["src_ids"]
        tgt_ids = batch["tgt_ids"]
        infill_mask = batch["infill_mask"]
        antigen_ids = batch["antigen_ids"]
        antigen_mask = batch["antigen_mask"]

        B = tgt_ids.size(0)
        t = torch.full((B,), self.T-1, device=self.device)
        x_t = self.q_sample(tgt_ids, t, infill_mask)

        attention_mask = (x_t != self.tokenizer.pad_token_id) & (~infill_mask.bool())
        h_ab = self.ab_encoder(x_t, attention_mask=attention_mask).last_hidden_state
        h_ag = self.ag_proj(self.ag_encoder(antigen_ids, attention_mask=antigen_mask).last_hidden_state)

        logits = self.diffusion(h_ab, h_ag, antigen_mask, t)
        preds = logits.argmax(-1)
        
        correct = ((preds == tgt_ids) & infill_mask).sum()
        total = max(infill_mask.sum(), 1)  
        aar = correct.float() / total.float()
        self.log("valid_AAR_epoch", aar, on_step=False, on_epoch=True, prog_bar=True, logger=True)
       
    # ===== add noise =====
    def q_sample(self, x0, t, mask):
        B, L = x0.shape
        x_t = x0.clone()
        beta_t = self.betas.to(x0.device)[t].view(B, 1)
        rand = torch.rand(B, L, device=x0.device)
        corrupt = (rand < beta_t) & mask  

        # random tokens from vocab
        noise = torch.randint(0, self.tokenizer.vocab_size, (B, L), device=x0.device)

        special_tokens = set(self.tokenizer.all_special_ids)
        for tok in special_tokens:
            corrupt &= (x0 != tok)

        x_t[corrupt] = noise[corrupt]
        return x_t
        
    # ===== sampling ===== 
    def sample(self, batch, n_samples):
        src = batch["src_ids"]
        infill_mask = batch["infill_mask"].bool()
        antigen_ids = batch["antigen_ids"]
        antigen_mask = batch["antigen_mask"]
        B, L = src.shape

        # ===== Greedy (deterministic) =====
        x_greedy = src.clone()
        for t in reversed(range(self.T)):
            attention_mask = (x_greedy != self.tokenizer.pad_token_id) & (~infill_mask)
            h_ab = self.ab_encoder(x_greedy, attention_mask=attention_mask).last_hidden_state
            h_ag = self.ag_proj(self.ag_encoder(antigen_ids, attention_mask=antigen_mask).last_hidden_state)
            logits = self.diffusion(h_ab, h_ag, antigen_mask, torch.full((B,), t, device=self.device))
            x_greedy[infill_mask] = logits.argmax(-1)[infill_mask]

        # ===== Stochastic samples =====
        samples = []
        i=1
        for k in range(n_samples):
            print(f"create sample_{i}")
            x = src.clone()
            for t in reversed(range(self.T)):
                attention_mask = (x != self.tokenizer.pad_token_id) & (~infill_mask)
                h_ab = self.ab_encoder(x, attention_mask=attention_mask).last_hidden_state
                h_ag = self.ag_proj(self.ag_encoder(antigen_ids, attention_mask=antigen_mask).last_hidden_state)
                logits = self.diffusion(h_ab, h_ag, antigen_mask, torch.full((B,), t, device=self.device))
                probs = F.softmax(logits/1.7, dim=-1)          
                sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, L)
                x[infill_mask] = sampled[infill_mask]

            samples.append(x.detach())
            i = i+1
        return x_greedy.detach(), samples

    # ===== test =====
    def test_step(self, batch, batch_idx):
        print(f"[test_step] batch {batch_idx} start")
        greedy, samples = self.sample(batch, self.num_samples)
        print(f"[test_step] batch {batch_idx} done")

        src_ids = batch["src_ids"]
        tgt_ids = batch["tgt_ids"]
        infill_mask = batch["infill_mask"]
        pdbs = batch["key"] 
       
        self.test_outputs.append({
            "pdbs": pdbs,
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "infill_mask": infill_mask,
            "samples": samples,                
            "greedy": greedy                   
        })
        
    def on_test_epoch_end(self):
        print(">>> Test epoch end started")
        esm = ESM2Scorer()
        succ_mean, tot_mean = 0, 0
        ppl_sum, ppl_cnt = 0.0, 0
        pdbs_all, src_all, tgt_all, best_all, greedy_all, div_all = [], [], [], [], [], []
        all_samples = []

        for out in self.test_outputs:
            src_ids = out["src_ids"]
            tgt_ids = out["tgt_ids"]
            infill_mask = out["infill_mask"]
            samples = out["samples"]
            greedy = out["greedy"]
            pdbs = out["pdbs"]

            B, K = src_ids.size(0), len(samples)
            for i in range(B):
                smpls_i = torch.stack([samples[k][i] for k in range(K)], 0)
                mask = infill_mask[i]

                # ===== DIV ===== 
                eq = smpls_i.unsqueeze(0) == smpls_i.unsqueeze(1)       
                m = mask.view(1, 1, -1)                                 

                sim = (eq * m).sum(dim=-1) / m.sum()                    
                div = 1.0 - sim.mean()                                  
                div_all.append(div.item())                              
    
                # ===== PPL =====
                infill_mask_i = infill_mask[i].repeat(self.num_samples, 1).to(self.device)
                ppl_scores = []
                n=1
                for s in smpls_i:
                    print(f"{i} ppl_{n}")
                    ppl_scores.append(esm.compute_pseudo_perplexity(s.unsqueeze(0), infill_mask_i[0:1], self.tokenizer))  
                    n = n+1
                    
                    succ_mean  += (s[mask] == tgt_ids[i][mask]).sum().item()
                    tot_mean  += mask.sum().item()

                best_idx = int(np.argmin(ppl_scores))
                best = smpls_i[best_idx]  
                greedy_i = greedy[i]
                
                ppl_sum += ppl_scores[best_idx]
                ppl_cnt += 1
                
                # ===== sequences =====
                best_full = tgt_ids[i].clone()
                best_full[mask] = best[mask]

                pred_full = tgt_ids[i].clone()
                pred_full[mask] = greedy_i[mask]

                pdbs_all.append(pdbs[i])
                src_all.append(decode_with_X(self.tokenizer, src_ids[i], mask))
                tgt_all.append(self.tokenizer.decode(tgt_ids[i], skip_special_tokens=True))
                best_all.append(self.tokenizer.decode(best_full, skip_special_tokens=True))
                greedy_all.append(self.tokenizer.decode(pred_full, skip_special_tokens=True))
                all_samples.append(self.tokenizer.batch_decode(smpls_i, skip_special_tokens=True))
        
        aar = 100 * succ_mean / tot_mean          
        div = 100 - 100 * np.mean(div_all)
        ppl = np.exp(ppl_sum / ppl_cnt)  
        
        root = self.trainer.default_root_dir
        print (">>> Write Fasta")
        def write_fasta(path, keys, seqs):
            with open(path, "w") as f:
                for k, s in zip(keys, seqs):
                    f.write(f">{k}\n{''.join(s.split())}\n")

        write_fasta(f"{root}/masked.fasta", pdbs_all, src_all)
        write_fasta(f"{root}/true.fasta", pdbs_all, tgt_all)
        write_fasta(f"{root}/pred_bstsmpl.fasta", pdbs_all, best_all)         
        write_fasta(f"{root}/pred.fasta", pdbs_all, greedy_all)             
        
        with open(f"{root}/pred_smpls.fasta", "w") as f:
            for pdb_id, smpls in zip(pdbs_all, all_samples):
                for i, s in enumerate(smpls, start=1):
                    f.write(f">{pdb_id}_{i}\n{''.join(s.split())}\n")

        results = {
            "AAR_test": aar,
            "PPL_test": ppl,       
            "DIV_test": div,          
            "num_samples": self.num_samples
        }
        
        log.info(f'AAR_test = {aar}, PPL_test = {ppl}, DIV_test = {div}, num_samples = {self.num_samples}')
        with open(f"{root}/results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(">>> Test finished:", results)

    # ===== optimizer =====
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.diffusion.parameters(), "lr": self.lr},
            {"params": self.ag_proj.parameters(), "lr": self.lr},
            {"params": self.ab_encoder.encoder.layer[-1].parameters(), "lr": self.lr * 0.1},
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
     
    # ===== inference =====
    def prepare_single_input(self, antibody_seq, antigen_seq):
        """
        antibody_seq: string with ***** for CDR-H3
        antigen_seq: raw antigen sequence
        """

        device = self.device

        # ===== Antibody =====
        tokens = []
        infill_mask = []

        for aa in antibody_seq:
            if aa == "*":
                tokens.append(self.tokenizer.mask_token)
                infill_mask.append(True)
            else:
                tokens.append(aa)
                infill_mask.append(False)

        ab_str = " ".join(tokens)
        ab_enc = self.tokenizer(ab_str, return_tensors="pt", add_special_tokens=True)

        src_ids = ab_enc["input_ids"].to(device)
        infill_mask = torch.tensor([False] + infill_mask + [False],device=device).unsqueeze(0)

        # ===== Antigen =====
        ag_enc = self.tokenizer(" ".join(list(antigen_seq)),return_tensors="pt",add_special_tokens=True)

        antigen_ids = ag_enc["input_ids"].to(device)
        antigen_mask = ag_enc["attention_mask"].to(device)

        return {
            "src_ids": src_ids,
            "infill_mask": infill_mask,
            "antigen_ids": antigen_ids,
            "antigen_mask": antigen_mask
        }

    @torch.no_grad()
    def generate(self, antibody_seq, antigen_seq, num_samples, temperature=1.7):
        self.eval()
        batch = self.prepare_single_input(antibody_seq, antigen_seq)

        src = batch["src_ids"]
        infill_mask = batch["infill_mask"]
        antigen_ids = batch["antigen_ids"]
        antigen_mask = batch["antigen_mask"]

        B, L = src.shape
        samples = []

        for k in range(num_samples):
            x = src.clone()
            for t in reversed(range(self.T)):
                attention_mask = (x != self.tokenizer.pad_token_id) & (~infill_mask)
                h_ab = self.ab_encoder(x, attention_mask=attention_mask).last_hidden_state
                h_ag = self.ag_proj(self.ag_encoder(antigen_ids, attention_mask=antigen_mask).last_hidden_state)

                logits = self.diffusion(h_ab, h_ag, antigen_mask,torch.full((B,), t, device=self.device))

                probs = F.softmax(logits / temperature, dim=-1)
                sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, L)

                x[infill_mask] = sampled[infill_mask]

            samples.append(x[0].clone())

        return samples, infill_mask[0]

    def generate_fasta(self, antibody_seq, antigen_seq, fasta_path, num_samples):
        samples, mask = self.generate(antibody_seq,antigen_seq,num_samples=num_samples)

        with open(fasta_path, "w") as f:
            for i, s in enumerate(samples, 1):
                seq = self.tokenizer.decode(s, skip_special_tokens=True)
                seq = "".join(seq.split())
                f.write(f">sample_{i}\n{seq}\n")


def decode_with_X(tokenizer, ids, mask):
    seq = tokenizer.decode(ids, skip_special_tokens=True)
    seq = list(seq.replace(" ", ""))  

    out = []
    j = 0
    for m in mask:
        if m:
            out.append("X")
        else:
            if j < len(seq):
                out.append(seq[j])
                j += 1

    return "".join(out)

def masked_mean_pool_ag(h, mask):
    mask = mask.unsqueeze(-1).float()
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    
def masked_mean_pool(h, mask):
    # h: [B, L, D]
    # mask: [B, L]
    mask = mask.unsqueeze(-1).float()
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    
