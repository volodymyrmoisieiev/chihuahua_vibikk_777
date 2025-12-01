from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

# These are optional dependencies installed via extras.
# If they are missing, import will fail with a clear message.
try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch import nn
    from torchvision import models, transforms
    import matplotlib.pyplot as plt  # new: for plotting
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        'Classification dependencies are missing. Install with: pip install -e ".[torch]"'
    ) from e


# Same class order you used in the notebook.
CLASS_NAMES: list[str] = ["chihuahua", "muffin"]

# Same preprocessing pipeline you used in the notebook.
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@dataclass(frozen=True)
class ClassificationResult:
    """Prediction output for the Chihuahua vs Muffin classifier."""

    label: str
    label_index: int
    probability: float


def _model_asset() -> object:
    """Return a Traversable pointing to bundled `best_model.pth` inside the package."""
    return files("chihuahua_tools").joinpath("assets", "models", "best_model.pth")


@lru_cache(maxsize=2)
def load_model(device: Optional[str] = None) -> "torch.nn.Module":
    """
    Load the ResNet18 model and its weights from package assets.

    Args:
        device: "cpu" or "cuda". If None, auto-selects.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18()
    num_classes = 2
    model.fc = nn.Linear(model.fc.infeatures, num_classes)  # type: ignore[attr-defined]

    # Load weights from packaged file.
    with as_file(_model_asset()) as model_path:
        state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def predict_image(
    image: "Image.Image",
    *,
    device: Optional[str] = None,
    model: Optional["torch.nn.Module"] = None,
) -> ClassificationResult:
    """
    Predict class for a PIL image.

    Args:
        image: PIL Image.
        device: "cpu" or "cuda". If None, auto-selects.
        model: Optional pre-loaded model (otherwise loaded from assets).

    Returns:
        ClassificationResult with top class and its probability.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = load_model(device)

    image_tensor = PREPROCESS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    probs = F.softmax(output[0], dim=0).detach().cpu()
    top_prob, top_class = torch.topk(probs, 1)

    label_index = int(top_class.item())
    probability = float(top_prob.item())
    label = CLASS_NAMES[label_index]

    return ClassificationResult(
        label=label,
        label_index=label_index,
        probability=probability,
    )


def predict_path(
    image_path: Union[str, Path],
    *,
    device: Optional[str] = None,
    model: Optional["torch.nn.Module"] = None,
) -> ClassificationResult:
    """
    Predict class for an image on disk.

    Args:
        image_path: Path to an image file.

    Returns:
        ClassificationResult.
    """
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    return predict_image(image, device=device, model=model)


def predict_many(
    image_paths: Iterable[Union[str, Path]],
    *,
    device: Optional[str] = None,
) -> list[ClassificationResult]:
    """
    Predict classes for multiple image paths.

    Note: model is loaded once and reused for speed.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    results: list[ClassificationResult] = []
    for p in image_paths:
        results.append(predict_path(p, device=device, model=model))
    return results


# --- New helper for plotting ---


def _plot_image_with_prediction(
    image: "Image.Image",
    result: ClassificationResult,
) -> None:
    """
    Show image with prediction title, similar to the test_setup notebook.
    """
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {result.label} ({result.probability * 100:.2f}%)")
    plt.show()


def predict_path_with_plot(
    image_path: Union[str, Path],
    *,
    device: Optional[str] = None,
    model: Optional["torch.nn.Module"] = None,
    show: bool = True,
) -> ClassificationResult:
    """
    Predict class for an image on disk and optionally show it with a matplotlib plot.

    Args:
        image_path: Path to an image file.
        device: "cpu" or "cuda". If None, auto-selects.
        model: Optional pre-loaded model (otherwise loaded from assets).
        show: If True, display the image with title "Predicted: ... (XX.XX%)".

    Returns:
        ClassificationResult.
    """
    path = Path(image_path)
    image = Image.open(path).convert("RGB")

    result = predict_image(image, device=device, model=model)

    if show:
        _plot_image_with_prediction(image, result)

    return result
