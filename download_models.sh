#!/bin/bash

# បង្កើត Folder នីមួយៗ
mkdir -p gender_model
mkdir -p age_model
mkdir -p emotion_model

echo "🤖 ប្រព័ន្ធចាប់ផ្តើមទាញយកម៉ូដែលបន្ថែម (Age & Emotion)..."

# ទាញយកឯកសារនីមួយៗ
echo "⏳ កំពុងទាញយក Gender Model..."
wget -q -O gender_model/gender_deploy.prototxt "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
wget -q -O gender_model/gender_net.caffemodel "https://github.com/arunponnusamy/cvlib-files/releases/download/v0.1/gender_net.caffemodel"

echo "⏳ កំពុងទាញយក Age Model..."
wget -q -O age_model/age_deploy.prototxt "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
wget -q -O age_model/age_net.caffemodel "https://huggingface.co/AjaySharma/genderDetection/resolve/main/age_net.caffemodel"

echo "⏳ កំពុងទាញយក Emotion Model..."
wget -q -O emotion_model/emotion_model.onnx "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

echo "✨ រួចរាល់ជោគជ័យទាំងអស់!"