# **Sentiment Analysis on Tweets Using LSTM**

This project focuses on classifying tweets into three sentiment categories: `bad`, `good`, and `neutral`. It uses a deep learning model built with LSTM (Long Short-Term Memory) layers to analyze and predict the sentiment of text data. The project includes data preprocessing, model training, and evaluation.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Structure](#code-structure)
6. [Results](#results)
7. [Dependencies](#dependencies)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

---

## **Overview**

The goal of this project is to build a text classification model that can predict the sentiment of tweets. The dataset contains tweets labeled as `bad`, `good`, or `neutral`. The model uses LSTM layers to capture sequential dependencies in the text data and classify the tweets into one of the three categories.

---

## **Features**

- **Data Preprocessing**:
  - Removes HTML tags and punctuation.
  - Converts text to lowercase.
  - Tokenizes and pads sequences for input to the model.
- **Model Architecture**:
  - Uses LSTM layers for sequential data processing.
  - Includes normalization and dense layers for classification.
- **Training and Evaluation**:
  - Trains the model using categorical cross-entropy loss and Adam optimizer.
  - Visualizes training accuracy and loss over epochs.
- **Improvements**:
  - Supports embedding layers, dropout, and bidirectional LSTM for better performance.

---

## **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/tweet-sentiment-analysis.git
   cd tweet-sentiment-analysis
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.x installed. Then, install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy Model** (if needed):
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## **Usage**

1. **Prepare the Dataset**:
   - Place your dataset (`file.csv`) in the project directory. The dataset should have two columns: `tweets` (text data) and `labels` (`bad`, `good`, `neutral`).

2. **Run the Script**:
   Execute the Python script to preprocess the data, train the model, and evaluate its performance:
   ```bash
   python sentiment_analysis.py
   ```

3. **Review the Output**:
   - The script will display training accuracy and loss plots.
   - The trained model will be saved for future use (optional).

---

## **Code Structure**

- **Data Preprocessing**:
  - Removes HTML tags and punctuation.
  - Tokenizes and pads sequences.
  - Encodes labels into numerical values.

- **Model Architecture**:
  - LSTM layers for sequential data processing.
  - Dense layers for classification.
  - Softmax activation for multi-class classification.

- **Training**:
  - Uses Adam optimizer and categorical cross-entropy loss.
  - Includes early stopping to prevent overfitting.

- **Visualization**:
  - Plots training accuracy and loss over epochs.

---

## **Results**

### **Training Accuracy and Loss**
- The model's training accuracy and loss are visualized over epochs.
- Example plots:
  - **Training Accuracy**:
    ![Training Accuracy](training_accuracy.png)
  - **Training Loss**:
    ![Training Loss](training_loss.png)

### **Model Performance**
- The model achieves high accuracy on the training set.
- Further evaluation (e.g., confusion matrix, classification report) can be added for detailed insights.

---

## **Dependencies**

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `tensorflow` (for LSTM model)
  - `scikit-learn` (for evaluation)
  - `BeautifulSoup` (for HTML tag removal)

---

## **Contributing**

Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.


---

## **Contact**

For questions or feedback, please contact:
- **SUCHISMITA DAS**
- **Email**: dassuchismita2020@gmail.com

---

This README provides a comprehensive guide to setting up and using the tweet sentiment analysis project. Follow the instructions to get started and explore the features of this powerful tool!
