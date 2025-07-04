# Visual Question Answering System

This project implements a **Visual Question Answering (VQA)** model that answers open-ended questions about an image.  
We used the **VizWiz dataset**, which contains real-world questions and answers by visually impaired users about images they captured.

---

## ✨ Overview
- Built an end-to-end VQA pipeline using deep learning and multimodal encodings.
- Encoded image-question pairs using the CLIP model.
- Trained a neural network to predict both the **answer** and its **type**.
- Evaluated the model on the VizWiz dataset and analyzed predictions.

---

## 🗂 Dataset
- **VizWiz-VQA Dataset**  
  Contains over 31,000 image-question-answer triplets collected in real-world scenarios by visually impaired users.
- Unique due to its real and noisy data, making it a good benchmark for practical VQA systems.

---

## 🛠 Technologies & Tools
- Python
- PyTorch
- CLIP (Contrastive Language-Image Pretraining)
- Google Colab
- Kaggle API (to fetch data)

---

## 🔗 Methodology
1. **Data Preprocessing**
   - Loaded the dataset using Kaggle API.
   - Selected the most confident answer for each question.
   - Encoded answers and types using one-hot encoding.
   - Balanced dataset by weighting answers by confidence.

2. **Feature Extraction**
   - Used OpenAI's CLIP to encode both image and question into joint embeddings.

3. **Model**
   - Designed a PyTorch neural network to predict:
     - The answer.
     - The answer type.
   - Combined predictions for final output.

4. **Training & Evaluation**
   - Trained on ~80 epochs.
   - Evaluated using separate validation and test splits.
   - Achieved:
     - Training accuracy ~82%
     - Test answer-type accuracy ~84%

---

## 📊 Results
```
| Metric                | Training | Validation | Test     |
|-----------------------|----------|------------|----------|
| Accuracy (overall)    | ~82%    | ~33%      | ~55%    |
| Answer Accuracy       | ~80%    | ~0%       | ~40%    |
| Answer Type Accuracy  | ~85%    | ~67%      | ~85%    |
```


