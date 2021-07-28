import torchmetrics as tm

metrics = [
    ['acc', tm.Accuracy(threshold=0.5, num_classes=2, average='micro')],
    ['f1', tm.F1(threshold=0.5, num_classes=2, average='micro')],
    ['precision', tm.Precision(threshold=0.5, num_classes=2, average='micro')],
    ['recall', tm.Recall(threshold=0.5, num_classes=2, average='micro')],
]