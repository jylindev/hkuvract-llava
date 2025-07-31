from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Union
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import PreTrainedTokenizer
from sklearn.metrics import classification_report,confusion_matrix
from sentence_transformers import SentenceTransformer, util
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from sklearn.metrics import precision_recall_fscore_support
import torch
import random
import hashlib



class ComputeMetrics:
    def __init__(self, tokenizer, sample_size=100):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        
    def _hash_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def compute(self, decoded_preds: list[str], decoded_labels: list[str]) -> Dict[str, float]:
        score_dict = {"rouge-1": [], "rouge-2": [], "bleu-4": []}

        # 抽样 500 条
        total = len(decoded_preds)
        if total > self.sample_size:
            sample_indices = random.sample(range(total), self.sample_size)
        else:
            sample_indices = list(range(total))
            
        preds_sample = decoded_preds
        labels_sample = decoded_labels

        # preds_sample = [decoded_preds[i] for i in sample_indices]
        # labels_sample = [decoded_labels[i] for i in sample_indices]

        # 用哈希 key 避免缓存文件名过长
        gts = {self._hash_key(str(i)): [labels_sample[i]] for i in range(len(labels_sample))}
        res = {self._hash_key(str(i)): [preds_sample[i]] for i in range(len(preds_sample))}
        
        rouge = Rouge()
        for pred, label in zip(preds_sample, labels_sample):
            pred_tokens = pred.split()
            label_tokens = label.split()

            if len(pred_tokens) == 0 or len(label_tokens) == 0:
                rouge_result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}}
            else:
                scores = rouge.get_scores(" ".join(pred_tokens), " ".join(label_tokens))
                rouge_result = {k: scores[0][k] for k in ["rouge-1", "rouge-2"]}

            for k, v in rouge_result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu(
                [label_tokens], pred_tokens,
                smoothing_function=SmoothingFunction().method3
            )
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        # 计算 CIDEr 和 SPICE
        cider_score, _ = Cider().compute_score(gts, res)
        spice_score, _ = Spice().compute_score(gts, res)
        
        score_dict["cider"] = [cider_score * 100]
        score_dict["spice"] = [spice_score * 100]

        return {k: float(sum(v) / len(v)) for k, v in score_dict.items()}


class ActionClass:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
        self.labels = [
            "Walking", "Running", "Jumping", "Bending down and up", "Standing", "Squatting",
            "Raising hand", "Shooting", "Waving hammer", "Throwing hammer", "Waving laser",
            "Cutting using laser", "Moving using controller", "Catching fishes using net",
            "Grabbing and collecting box", "Measuring length","Waving sword",
            "Bowling", "Picking and Placing"
        ]
        
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        self.embedder = SentenceTransformer("/root/autodl-tmp/bge-large-en-v1.5")
        self.label_embs = self.embedder.encode(self.labels, convert_to_tensor=True)
        
        
    def compute(self, decoded_preds: list[str], decoded_labels: list[str]) -> Dict[str, str]:
        # decoded_preds = [pred.split('.')[0].strip() for pred in decoded_preds]
        decoded_labels = [label.split('.')[0].strip() for label in decoded_labels]
        
        pred_class_ids = []
        label_class_ids = []

        pred_embs = self.embedder.encode(decoded_preds, convert_to_tensor=True)
        
        label_embs = self.embedder.encode(decoded_labels, convert_to_tensor=True)
        
        top1_correct = 0
        top3_correct = 0
        
        for i in range(len(decoded_preds)):
            pred_sim = util.cos_sim(pred_embs[i], self.label_embs)[0]
            label_sim = util.cos_sim(label_embs[i], self.label_embs)[0]

            pred_cls = pred_sim.argmax().item()
            label_cls = label_sim.argmax().item()
            
            pred_class_ids.append(pred_cls)
            label_class_ids.append(label_cls)

            topk = torch.topk(pred_sim, k=3).indices.tolist()
            if label_cls == topk[0]:
                top1_correct += 1
            if label_cls in topk:
                top3_correct += 1
    
        pred_class_ids = np.array(pred_class_ids)
        label_class_ids = np.array(label_class_ids)
        per_class_result = {}
        for class_id in range(len(self.labels)):
            true_mask = (label_class_ids == class_id)
            total = true_mask.sum()
            
            correct = (pred_class_ids[true_mask] == class_id).sum()
            acc = correct / total * 100
            per_class_result[self.id2label[class_id]] = f"{correct} / {total} ({acc:.2f}%)"
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_class_ids, pred_class_ids, average='macro', zero_division=0
        )

        # 计算混淆矩阵
        cm = confusion_matrix(label_class_ids, pred_class_ids, labels=np.arange(len(self.labels)))
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # 行归一化
        cm_normalized = np.round(cm_normalized, 2)

        results = {
            "top1_accuracy": f"{top1_correct} / {len(decoded_preds)} ({top1_correct / len(decoded_preds) * 100:.2f}%)",
            "top3_accuracy": f"{top3_correct} / {len(decoded_preds)} ({top3_correct / len(decoded_preds) * 100:.2f}%)",
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "f1_score": f"{f1:.4f}",
            "per_class_accuracy": per_class_result,
            "confusion_matrix": cm.tolist(),  # 原始数量
            "confusion_matrix_normalized": cm_normalized.tolist()  # 归一化矩阵
        }
        
        return results