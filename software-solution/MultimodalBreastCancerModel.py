import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel

class MultimodalBreastCancerModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3,classifier = "linear"):
        super(MultimodalBreastCancerModel, self).__init__()


        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.image_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # ResNet-50 -> 2048-dimensional vector
        self.image_embed_size = resnet.fc.in_features


        self.text_extractor = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # BERT -> 768-dimensional vector
        self.text_embed_size = self.text_extractor.config.hidden_size


        # 2048 + 768 = 2816 dimensions
        combined_size = self.image_embed_size + self.text_embed_size

        # Dropout to randomly zero-out elements and prevent overfitting
        self.dropout = nn.Dropout(p=dropout_rate)

        # classification layer
        if classifier == "linear":
            self.classifier = nn.Linear(combined_size, num_classes)

        elif classifier == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(combined_size, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes)
            )

    def forward(self, images, input_ids, attention_mask):
        img_features = self.image_extractor(images)
        img_features = torch.flatten(img_features, 1)

        text_outputs = self.text_extractor(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output


        fused_features = torch.cat((img_features, text_features), dim=1)

        x = self.dropout(fused_features)
        logits = self.classifier(x)

        return logits