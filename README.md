# AI Video Consultation Backend

Backend for the AI Video Consultation project built with Django and Django REST Framework.

## Prerequisites

- Python 3.7+  
- Git  
- Virtual environment (recommended)

## Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/Masa-Beydoun/AI-Video-Consultation-Backend.git
cd AI-Video-Consultation-Backend
````

2. **Create and activate a virtual environment (recommended):**

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run migrations:**

```bash
python manage.py migrate
```

5. **Start the server:**

```bash
python manage.py runserver
```

6. **Access the API:**

* Default endpoint: `http://127.0.0.1:8000/api/`

## Dependencies

* Django >= 3.2
* djangorestframework >= 3.12
* djangorestframework-simplejwt >= 5.0.0
* django\_filters
* numpy
* pydub
* librosa
* openai-whisper
* firebase-admin
* python-dotenv
* opencv-python
