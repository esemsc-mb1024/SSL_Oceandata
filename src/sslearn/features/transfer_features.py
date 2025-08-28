import torch

def extract_features_Sim(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            feats = model.encoder(images)
            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    return features, labels

def extract_features_dino(model, dataloader, device):
    """
    Feature extractor for DINO models.
    Returns CLS embeddings (pre-projector) if `return_features=True` is supported.
    """
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            feats = model(images, return_features=True)  # <- ensures CLS embeddings
            all_features.append(feats.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    return features, labels
