import timm
import torch
import torch.nn as nn


class MultimodalFusionModel(nn.Module):

  def __init__(self, model_name='efficientnet_b0', num_tabular_features=2):
    super(MultimodalFusionModel, self).__init__()

    # 1. Image Branch (Pre-trained Backbone)
    # num_classes=0 strips the final classification layer, returning raw feature embeddings
    self.image_encoder = timm.create_model(
        model_name, pretrained=True, num_classes=0
    )
    img_feature_dim = self.image_encoder.num_features

    # 2. Tabular Branch (Simple MLP)
    self.tabular_encoder = nn.Sequential(
        nn.Linear(num_tabular_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
    )

    # 3. Fusion & Classification Head
    fusion_input_dim = img_feature_dim + 16
    self.classifier = nn.Sequential(
        nn.Linear(fusion_input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),  # Outputs a single logit for binary classification
    )

  def forward(self, images, tabular):
    # Extract features
    img_feats = self.image_encoder(images)
    tab_feats = self.tabular_encoder(tabular)

    # Concatenate features together
    combined = torch.cat((img_feats, tab_feats), dim=1)

    # Pass through final classifier head
    out = self.classifier(combined)
    return out.squeeze(1)