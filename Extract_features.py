def extract_features(model, dataloader, device):
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