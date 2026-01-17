from model import SimpleCNN


def build_cnn(
    num_classes,
    embedding_dim,
    dropout,
    activation,
    image_size,
):
    return SimpleCNN(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout=dropout,
        activation=activation,
        image_size=image_size,
    )
