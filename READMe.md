# ព័ត៌មានអំពី Web និង AI ដែលបានប្រើ

## ភាសាដែល Web នេះប្រើ

- **Python (Flask backend)** — `app.py`
- **HTML** — `index.html`
- **CSS** — `style.css`
- **JavaScript (client-side, inline ក្នុង index.html)** — `index.html`
- **Shell / Bash** — `download_models.sh`
- **ម៉ូឌែល/ផែនការទិន្នន័យ**: Caffe (`.prototxt`, `.caffemodel`) និង ONNX (`.onnx`)

---

## AI និងម៉ូឌែលដែលបានប្រើ

### ស្គាល់មុខ (Face Detection / Recognition)
- បណ្ណាល័យ Python `face_recognition` (ផ្អែកលើ dlib)
- ប្រើសម្រាប់ encoding និងប្រៀបធៀបមុខ

### កំណត់ភេទ និងអាយុ (Gender / Age Detection)
- OpenCV DNN
- ម៉ូឌែលនៅក្នុងថត `gender_model` និង `age_model`
- ប្រើឯកសារ `*.prototxt` និង `*.caffemodel`

### កំណត់អារម្មណ៍ (Emotion Detection)
- OpenCV DNN
- ប្រើម៉ូឌែល `emotion_model.onnx`
- មាន fallback heuristics សម្រាប់ពិចារណា landmark

### បណ្ណាល័យបន្ថែម
- `cv2` (OpenCV) សម្រាប់ preprocessing និង inference
- `dlib` ជាការពឹងផ្អែកក្រោម `face_recognition`

### ចំណាំ
- ក្នុង virtual environment មាន `deepface`
- ប៉ុន្តែ `app.py` ប្រើ `face_recognition` + OpenCV DNN ជាចម្បង
- មិនឃើញមានការប្រើ `deepface` ដោយផ្ទាល់ក្នុង script

---

## ម៉ូឌែលដែល Download ពី GitHub

### 1. gender_deploy.prototxt
URL:
https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt

Repository:
spmallick/learnopencv

### 2. gender_net.caffemodel
URL:
https://github.com/arunponnusamy/cvlib-files/releases/download/v0.1/gender_net.caffemodel

Repository:
arunponnusamy/cvlib-files

### 3. age_deploy.prototxt
URL:
https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt

Repository:
spmallick/learnopencv

### 4. emotion_model.onnx
URL:
https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx

Repository:
onnx/models

### ម៉ូឌែលដែលមិនមកពី GitHub

#### age_net.caffemodel
URL:
https://huggingface.co/AjaySharma/genderDetection/resolve/main/age_net.caffemodel

Source:
Hugging Face

---

# ជំហាន Run Web Application

## 1. ចូល Virtual Environment

```bash
source env/bin/activate
```

## 2. តម្លើង Dependencies

```bash
pip install -r requirements.txt
```

## 3. Download ម៉ូឌែល

```bash
bash download_models.sh
```

បើ `wget` មិនទាន់មាន:

```bash
sudo apt install wget
```

(Debian/Ubuntu)

## 4. Run Development Server

```bash
python app.py
```

បន្ទាប់មកបើក Browser ទៅ:

```text
http://localhost:5000
```

---

## Production Mode (Gunicorn)

```bash
env/bin/gunicorn -k gthread --threads 4 -w 1 -b 0.0.0.0:8000 app:app
```

ឬ

```bash
PORT=8000 python app.py
```

---

# ការដោះស្រាយបញ្ហា

## Model Files Not Found

ពិនិត្យថាម៉ូឌែលមាននៅក្នុងថត:

```bash
ls gender_model age_model emotion_model
```

## Camera មិនដំណើរការ

ពិនិត្យ៖

- video0 device មានឬអត់
- User មានសិទ្ធិប្រើ Camera ឬអត់

## Python Library Errors

ដំណើរការ៖

```bash
pip install -r requirements.txt
```

ហើយពិនិត្យថាកំពុងប្រើ Python Environment ត្រឹមត្រូវ។

---

## ចំណាំ

`app.py` ត្រូវបានកំណត់ឲ្យ Run Flask Development Server ដោយផ្ទាល់ តាមរយៈ៖

```bash
python app.py
```

