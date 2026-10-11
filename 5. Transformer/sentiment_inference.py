"""From the notebook after training:

from sentiment_inference import predict_sentiment
predict_sentiment(model, classifier_tokenizer, "The food was wonderful!")
"""

import torch


@torch.inference_mode()
def predict_sentiment(model, tokenizer, text):
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=model.embed.pos.num_embeddings,
        return_tensors="pt",
    )
    model.eval()
    logit = model(inputs["input_ids"].to(device))[0]
    positive = torch.sigmoid(logit).item()
    return {
        "label": "positive" if positive >= 0.5 else "negative",
        "negative": 1 - positive,
        "positive": positive,
    }
