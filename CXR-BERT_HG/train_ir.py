"""
Downstream task : Image Retrieval
    - I_a, T_a -> pair
    - I_a, T_0 ~ T_19 -> negative samples
    - [CLS] -> softmax cross-entropy

TODO
1. load pre-trained model
2. random data sampling ( pair, negative samples)
3. train, validation
4. inference
    - Metric: Recall@1, 5, 10
"""