import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from networks.resnet import get_model
from data.datasets import get_dataloader
from sklearn.metrics import accuracy_score, average_precision_score

TEST_DIR = "images/test"
BATCH_SIZE = 4
MODEL_PATH = "models/model.pth"


def getLabel(label_code):
    return 'Real' if label_code == 1 else 'Fake'


def evaluate():
    test_loader, _ = get_dataloader(
        TEST_DIR, batch_size=BATCH_SIZE, shuffle=True)

    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Evaluating"):
            output = model(images)
            probs = torch.sigmoid(output).squeeze(1)

            preds = (probs > 0.5).int()

            for i in range(len(labels)):
                actual = labels[i].item()
                pred = preds[i].item()

                if pred != actual:
                    img = Image.open(paths[i]).convert("RGB")
                    plt.imshow(img)
                    plt.title(
                        f"Actual: {getLabel(labels[i].item())}, Pred: {getLabel(preds[i].item())}")
                    plt.axis("off")
                    plt.show()

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    predictions = (all_probs > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    ap = average_precision_score(all_labels, all_probs)

    print(f"Test Accuracy: {accuracy*100:.1f}%")
    print(f"Average Precision: {ap*100:.1f}%")


if __name__ == "__main__":
    evaluate()
