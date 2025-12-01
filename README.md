# chihuahua-tools 🐶🧰

This project is a simple Python toolkit with several fun Chihuahua-themed features.  
It is made for learning and experimenting with Python, machine learning, and computer vision.

The toolkit includes:

- **Behavior / vibe model** (scikit-learn)
- **Chihuahua vs Muffin classifier** (PyTorch)
- **Emoji webcam demo** (OpenCV + MediaPipe)
- **Horoscope generator** (OpenAI API)

---

## Requirements

- Python: **3.10**, **3.11**, **3.12**
- Recommended: **Python 3.11**
- Not recommended: **Python 3.13** (because some libraries like Mediapipe cannot install correctly)

---

## Installation (developer mode)

Clone the repository and install the package:

```bash
git clone https://github.com/volodymyrmoisieiev/chihuahua_vibikk_777.git
cd chihuahua_vibikk_777

pip install -e .
```

### Optional extras

You can install only the parts that you need:

```bash
pip install -e ".[torch]"      # for PyTorch classifier
pip install -e ".[cv]"         # for webcam emoji demo
pip install -e ".[horoscope]"  # for OpenAI horoscope tool
pip install -e ".[dev]"        # for development tools
```

---

## Project structure

- `src/chihuahua_tools/` – main package code  
- `src/chihuahua_tools/assets/` – models and images  
- `notebooks/` – Jupyter notebooks with experiments  
- `tests/` – test files (if added in the future)

Some functions load files from:

- `assets/models/`
- `assets/images/`

Please keep the structure unchanged unless you update the code that loads these files.

---

## Examples

### 1. Behavior / vibe model

This model takes simple binary features and predicts a “Chihuahua vibe”.

```python
from chihuahua_tools.behavior import predict_chi_vibe

features = {
    "Не гавкає 24/7": 1,
    "Не гризе взуття": 1,
    "Шарить в таро та астрології": 1,
    "Цінує СМП, ML та DL": 1,
    "Кайфує від Taylor Swift": 1,
    "Їсть суші": 1,
    "Шанує IQOS культуру": 1,
}

result = predict_chi_vibe(features)

print("Prediction:", result.prediction)
print("Probability:", result.probability)
print("Text:", result.text)
```

The function returns a result object with attributes such as:
- `prediction`
- `probability`
- `text`

---

### 2. Chihuahua vs Muffin classifier (PyTorch)

Install the extra first:

```bash
pip install -e ".[torch]"
```

Then classify an image:

```python
from chihuahua_tools.classification import predict_path_with_plot

result = predict_path_with_plot("path/to/image.jpg")
print(result.label, result.probability)
```

❗ **Important:**  
`predict_path()` returns a **single object**, not a tuple.  
Do **not** write:

```python
label, prob = predict_path("image.jpg")  # ❌ This will give an error
```

Always use the `result` object.

---

### 3. Emoji webcam demo

Install dependencies:

```bash
pip install -e ".[cv]"
```

Run the demo:

```python
from chihuahua_tools.emoji import main

if __name__ == "__main__":
    main()
```

This demo opens your webcam and shows simple emoji effects based on your face movements.

---

### 4. Horoscope generator (OpenAI API)

Install dependencies:

```bash
pip install -e ".[horoscope]"
```

Create a `.env` file with your API key:

```env
OPENAI_API_KEY=your_key_here
```

Use the generator:

```python
from chihuahua_tools.horoscope import generate_chihuahua_horoscope

text = generate_chihuahua_horoscope(
    name="Sasha",
    details="Loves Converse and say the phrase OHHH MY FUCKING GOD",
)

print(text)
```

---

## Development

If you want to develop or change the project:

```bash
pip install -e ".[dev]"
```

Then you can run:

```bash
pytest      # to run tests
ruff check  # to lint code
black .     # to format code
```

---

## How to Run (Beginner Friendly Guide)

This section explains how to run the project even if you are new to programming.

### 1. Install Python

Download Python 3.11 from:

https://www.python.org/downloads/release/python-3118/

During installation check:

```
[✔] Add Python to PATH
```

### 2. Create a project folder

Example:

```
C:\\Users\\YourName\\Desktop\\chihuahua_test\\
```

### 3. Open the folder in VS Code

If you do not have VS Code: https://code.visualstudio.com/

Open VS Code → File → Open Folder → select `chihuahua_test`.

### 4. Create a virtual environment

Open Terminal:

```bash
cd C:\\Users\\YourName\\Desktop\\chihuahua_test
python -m venv .venv
```

Activate it:

```bash
.\.venv\Scripts\activate
```

### 5. Install the package

Basic installation:

```bash
pip install "chihuahua-tools @ git+https://github.com/volodymyrmoisieiev/chihuahua_vibikk_777.git"
```

With PyTorch:

```bash
pip install "chihuahua-tools[torch] @ git+https://github.com/volodymyrmoisieiev/chihuahua_vibikk_777.git"
```

Everything:

```bash
pip install "chihuahua-tools[torch,cv,horoscope] @ git+https://github.com/volodymyrmoisieiev/chihuahua_vibikk_777.git"
```

### 6. Create a simple test script

Create:

```
solution.py
```

Add:

```python
from pathlib import Path
from chihuahua_tools.classification import predict_path_with_plot

BASE_DIR = Path(__file__).resolve().parent
image_path = BASE_DIR / "my_image.jpg"

result = predict_path_with_plot(str(image_path))
print(result.label, result.probability)
```

Place `my_image.jpg` next to `solution.py`.

### 7. Run the script

```
python solution.py
```

You will see the image window and prediction.

### 8. Update to newest version

```bash
.\.venv\Scripts\activate
pip uninstall chihuahua-tools -y
pip install "chihuahua-tools @ git+https://github.com/volodymyrmoisieiev/chihuahua_vibikk_777.git"
```

---

## License

This project uses the **MIT License**.  
See the `LICENSE` file for more details.
