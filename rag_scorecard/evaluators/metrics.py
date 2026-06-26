from typing import List
from rag_scorecard.models import RAGSample, MetricResult

def is_relevant(context: str, ground_truth: str, overlap_threshold: float = 0.2) -> bool:
    if not ground_truth or not context: 
        return False
        
    stop_words = {'the', 'a', 'an', 'is', 'are', 'to', 'if', 'in', 'of', 'and', 'or', 'has', 'out', 'for'}
    
    # Strip punctuation and tokenize to track baseline intersection
    ctx_words = set(w.strip(".,()?!\"'") for w in context.lower().split() if w not in stop_words)
    gt_words = set(w.strip(".,()?!\"'") for w in ground_truth.lower().split() if w not in stop_words)
    
    if not gt_words: 
        return False
        
    intersection = ctx_words.intersection(gt_words)
    overlap_ratio = len(intersection) / len(gt_words)
    
    return overlap_ratio >= overlap_threshold

def calculate_mrr(samples: List[RAGSample]) -> MetricResult:
    if not samples:
        return MetricResult(metric_name="MRR", score=0.0)
        
    reciprocal_ranks = []
    for sample in samples:
        # If ground_truth is missing/empty, it's an automatic 0 rank hit, but still a sample!
        if not sample.ground_truth:
            reciprocal_ranks.append(0.0)
            continue
            
        rank = 0
        for i, ctx in enumerate(sample.retrieved_contexts):
            if is_relevant(ctx, sample.ground_truth):
                rank = i + 1
                break
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
        
    score = sum(reciprocal_ranks) / len(samples)
    return MetricResult(metric_name="MRR", score=score)

def calculate_hit_rate(samples: List[RAGSample], k: int = 5) -> MetricResult:
    if not samples:
        return MetricResult(metric_name=f"Hit Rate @ {k}", score=0.0)
        
    hits = 0
    for sample in samples:
        if not sample.ground_truth:
            continue
        if any(is_relevant(ctx, sample.ground_truth) for ctx in sample.retrieved_contexts[:k]):
            hits += 1
            
    score = hits / len(samples)
    return MetricResult(metric_name=f"Hit Rate @ {k}", score=score)
