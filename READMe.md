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

---

# ដាក់នៅលើ Cloud (GitHub → Render / Heroku / Railway)

## ដំណើរការ GitHub ពីលើ

វេបនេះបានផ្ទុកលើ GitHub រួចហើយ។ ក្នុងការដាក់នៅលើ cloud host, តាមដានជំហាននេះ៖

### 1. Clone Repository ពីលើម៉ាស៊ីនលើក្រោយ ឬលើ Cloud Host

```bash
git clone https://github.com/Kane-kiraa/FaceDetectionProject.git
cd FaceDetectionProject
```

### 2. ដាក់នៅលើ Render (ល្អបំផុត)

**ជំហាន A: Sign up នៅ Render**
- ចូល https://render.com
- Sign up ប្រើ GitHub account

**ជំហាន B: Connect Repository**
1. ចូលទៅ "Dashboard" > "New +" > "Web Service"
2. Connect GitHub repo: `FaceDetectionProject`
3. កំណត់ settings:
   - **Name**: `facescanner` (ឬឈ្មោះផ្សេងទៀត)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && bash download_models.sh`
   - **Start Command**: `gunicorn -k gthread --threads 4 -w 1 -b 0.0.0.0:$PORT app:app`
   - **Instance Type**: `Standard` (CPU, RAM រឹង)
4. Deploy!

**ចំណាំ**: ម៉ូឌែលធំ (caffemodel, onnx) អាចជ្រើង 30-60 វិនាទីក្នុងការ download នៅលើ build វេលា។

### 3. ដាក់នៅលើ Heroku (បានបាត់សមតាភាព - ឃ្លាឈ្មោះលែងសម្រាប់ free tier)

Heroku ឥឡូវលែងផ្តល់ free tier ដូចដើម (ចាប់ពីឆ្នាំ 2022)។ ប្រសិនបើមាន Heroku account, ដាក់ `Procfile` នៅ root (ដែលបានបង្កើតរួចហើយ):

```bash
# Procfile (មាននៅក្នុង root already)
web: gunicorn -k gthread --threads 4 -w 1 -b 0.0.0.0:$PORT app:app
```

បន្ទាប់មក:
```bash
heroku login
heroku create <app-name>
git push heroku main
```

### 4. ដាក់នៅលើ Railway

1. ចូល https://railway.app
2. "New Project" > Connect GitHub
3. Select `FaceDetectionProject` repository
4. Railway នឹង detect `Procfile` និង deploy ដោយស្វ័យប្រវត្ត

### 5. ដាក់នៅលើម៉ាស៊ីនផ្ទាល់ខ្លួន (VPS / Linux Server)

```bash
# SSH ទៅម៉ាស៊ីនលើក្រោយ
ssh user@your-server-ip

# Clone repository
git clone https://github.com/Kane-kiraa/FaceDetectionProject.git
cd FaceDetectionProject

# ដាក់ virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
bash download_models.sh

# Run ក្រោយ Gunicorn + Nginx
gunicorn -k gthread --threads 4 -w 1 -b 127.0.0.1:8000 app:app &

# (Optional) Setup Nginx reverse proxy:
# /etc/nginx/sites-available/default
# server {
#     listen 80;
#     server_name your-domain.com;
#     location / {
#         proxy_pass http://127.0.0.1:8000;
#     }
# }
```

---

## ⚠️ សំខាន់៖ Camera Limitations នៅលើ Cloud

- **WebCam / Local Camera (`/dev/video*`)**: មិនអាច access ពីលើ cloud host
- **ដំណោះស្រាយ**:
  1. **IP Camera / RTSP Stream**: ប្រើ IP camera ឬ stream RTSP (ក្នុង `app.py` ប្ដូរ `source` ទៅ `"rtsp://..."`)
  2. **Local Machine**: រត់វេបលើម៉ាស៊ីនដែលមាន camera (`python app.py` or Gunicorn locally)
  3. **WebRTC/Stream from Client**: ដាក់ JavaScript ក្នុង browser ដើម្បីផ្ញើ video stream ទៅម៉ាស៊ីនលើក្រោយ

---

## 🔗 Repository Links

- **GitHub**: https://github.com/Kane-kiraa/FaceDetectionProject
- **Live Demo** (នឹងរត់នៅលើ Render/Heroku ប្រសិនបើដាក់រួច)

---

## តម្លើង Dependencies សម្រាប់ Cloud (requirements.txt)

សូមធានាថា `requirements.txt` មាន:

```txt
Flask==3.1.3
Flask-CORS==6.0.2
opencv-python-headless==4.10.0.84
face-recognition==1.3.0
dlib==20.0.1
numpy==1.26.0
gunicorn==26.0.0
```

ប្រសិនបើចាប់ផ្តើម pip install គ្មាន `gunicorn`, ដាក់វា៖

```bash
echo "gunicorn==26.0.0" >> requirements.txt
pip install -r requirements.txt
```

