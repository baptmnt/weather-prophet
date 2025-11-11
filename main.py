

if __name__ == "__main__":
    from training.train import train_model
    from models.simple_1img import SimpleCNN
    from torch.utils.data import TensorDataset
    import torch

    # Données factices
    N = 1000
    X = torch.randn(N, 3, 28, 28)
    y = torch.randn(N, 1)

    dataset = TensorDataset(X, y)

    model = SimpleCNN()

    params = [
        [10, 32, 1e-3],
        [32, 64, 1e-3],
        [64, 128, 1e-3],
    ]
    for p in params:
        train_model("simple_cnn", model, dataset, num_epochs=p[0], batch_size=p[1], learning_rate=p[2])