# chihuahua-tools 🐶🧰

A small Python toolkit with several “Chihuahua” mini-projects:

- **Behavior / vibe model** (scikit-learn)
- **Chihuahua vs Muffin classifier** (PyTorch)
- **Emoji webcam demo** (OpenCV + MediaPipe)
- **Horoscope generator** (OpenAI API)

This is a **learning project**.

---

## Requirements

- **Python:** 3.10 / 3.11 / 3.12  
  (Python 3.13 is not supported because `mediapipe` may not install.)
- Recommended: **Python 3.11**

---

## Installation (developer mode)

Clone the repository and install the package in editable mode:

```bash
pip install -e .
```

### Install optional modules (extras)

Install only what you need:

```bash
# Chihuahua vs Muffin (PyTorch)
pip install -e ".[torch]"

# Emoji webcam demo (OpenCV + MediaPipe)
pip install -e ".[cv]"

# Horoscope generator (OpenAI + dotenv)
pip install -e ".[horoscope]"

# Tests (optional)
pip install -e ".[dev]"
```

---

## Project structure (short)

- `src/chihuahua_tools/` – package source code
- `src/chihuahua_tools/assets/` – bundled assets (models, images)
- `notebooks/` – Jupyter notebooks (experiments)
- `tests/` – tests (optional)

---

## Usage examples

> Note: module names depend on your folder structure inside `src/chihuahua_tools/`.
> If something does not import, check the exact package/module names.

### 1) Behavior / vibe model

```python
from chihuahua_tools.behavior import predict_chi_vibe

features = {
    "Не гавкає 24/7": 1,
    "Їсть суші": 1,
}

result = predict_chi_vibe(features)
print(result.prediction, result.probability, result.text)
```

### 2) Chihuahua vs Muffin classifier (PyTorch)

Install first:

```bash
pip install -e ".[torch]"
```

Example:

```python
from chihuahua_tools.classification import predict_path

result = predict_path("path/to/image.jpg")
print(result.label, result.probability) 
```

### 3) Emoji webcam demo (OpenCV + MediaPipe)

Install first:

```bash
pip install -e ".[cv]"
```

Run:

```python
from chihuahua_tools.emoji import main
main()
```

### 4) Horoscope generator (OpenAI API)

Install first:

```bash
pip install -e ".[horoscope]"
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here
```

Example:

```python
from chihuahua_tools.horoscope import generate_chihuahua_horoscope

text = generate_chihuahua_horoscope("Fonya", details="Loves sushi")
print(text)
```

---

## Notes about model/assets

Some modules use local files from:

- `src/chihuahua_tools/assets/models/`
- `src/chihuahua_tools/assets/images/`

If you move or rename assets, update the code that loads them.

---

## License

See `LICENSE`.
