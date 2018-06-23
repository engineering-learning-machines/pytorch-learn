# Transfer Learning Results

## PyTorch Transfer Learning Tutorial

Model: ResNet50 with an dense linear layer (nn.Linear)
Weights: Only the last layer's weights get modified
Dataset: Dogs and Cats
Parameters:

| Parameter | Value | Description |
| -- | --- | ---- |
| Classes | 2 | names: dogs, cats |
| Training Dataset Size | 23000 ||
| Batch Size | 16 |
| Number of Epochs | 25 |
| Validation Dataset Size | 2000 ||
| Optimizer | SGD with momentum|optim.SGD|
| Learning rate| 0.001||
| Momentum | 0.9 ||
| Learning Rate Scheduler|
| Decay Rate|0.1|lr_scheduler.StepLR|
| Decay Interval|7|Every 7 epochs|

### Result

Accuracy: 0.989500
Total Training Time: 35m 37s