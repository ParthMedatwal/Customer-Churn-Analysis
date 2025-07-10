# 📊 Customer Churn Analysis & Prediction

A comprehensive data science project that analyzes customer churn patterns and builds predictive models to identify customers at risk of leaving. The project includes exploratory data analysis, machine learning model development, and a deployable application for real-time predictions.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Pipeline](#analysis-pipeline)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Customer churn analysis is critical for businesses to understand why customers leave and predict which customers are at risk. This project provides a complete end-to-end solution including:

- **Exploratory Data Analysis (EDA)** to understand churn patterns
- **Feature Engineering** and data preprocessing
- **Machine Learning Models** for churn prediction
- **Model Evaluation** and performance analysis
- **Deployable Application** for real-time predictions

## ✨ Features

- 🔍 **Comprehensive EDA**: Deep dive into customer behavior and churn patterns
- 📈 **Advanced Analytics**: Statistical analysis and correlation studies
- 🤖 **Machine Learning Pipeline**: Multiple algorithms for churn prediction
- 💾 **Model Persistence**: Saved trained models for deployment
- 🚀 **Production Ready**: Deployable application with `app.py`
- 📊 **Rich Visualizations**: Interactive plots and insights
- 📄 **Detailed Documentation**: Complete analysis report in PDF format

## 📂 Dataset

The project analyzes customer data including:
- **Customer Demographics**: Age, gender, location, etc.
- **Service Information**: Plans, features, tenure
- **Usage Patterns**: Call details, data usage, billing
- **Churn Status**: Target variable for prediction

### Data Files:
- `first_telc.csv` - Primary dataset
- `tel_churn.csv` - Additional churn data
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Comprehensive customer dataset

## 📁 Project Structure

```
customer-churn-analysis/
│
├── .ipynb_checkpoints/                      # Jupyter checkpoint files
├── images/                                  # Visualization outputs
├── templates/                               # HTML templates for web app
├── app.py                                   # Main Flask/Streamlit application (5 KB)
├── Churn Analysis - EDA.ipynb             # Exploratory Data Analysis notebook (875 KB)
├── Churn Analysis - Model Building.ipynb  # Machine Learning model development (61 KB)
├── first_telc.csv                          # Primary dataset (10 KB)
├── model.sav                               # Trained model (pickled) (577 KB)
├── README.md                               # Project documentation (3 KB)
├── tel_churn.csv                           # Secondary churn dataset (801 KB)
├── Untitled.ipynb                          # Additional analysis notebook (9 KB)
└── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Main customer churn dataset (955 KB)
```

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Clone the Repository
```bash
git clone https://github.com/ParthMedatwal/customer-churn-analysis.git
cd customer-churn-analysis
```

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter flask streamlit plotly
```

### For Jupyter Notebooks
```bash
pip install jupyter notebook ipykernel
```

## 💻 Usage

### 1. Run the Web Application
```bash
python app.py
```

### 2. Explore the Analysis
```bash
jupyter notebook "Churn Analysis - EDA.ipynb"
```

### 3. Review Model Building
```bash
jupyter notebook "Churn Analysis - Model Building.ipynb"
```

### 4. Load Trained Model
```python
import pickle
model = pickle.load(open('model.sav', 'rb'))
# Make predictions on new data
predictions = model.predict(new_customer_data)
```

## 🔄 Analysis Pipeline

### Phase 1: Exploratory Data Analysis
- **Data Quality Assessment**: Missing values, duplicates, data types
- **Univariate Analysis**: Distribution of individual features
- **Bivariate Analysis**: Relationship between features and churn
- **Multivariate Analysis**: Complex feature interactions
- **Statistical Testing**: Hypothesis testing for significance

### Phase 2: Data Preprocessing
- **Data Cleaning**: Handle missing values and outliers
- **Feature Engineering**: Create new meaningful features
- **Encoding**: Convert categorical variables to numerical
- **Scaling**: Normalize features for model training
- **Feature Selection**: Select most relevant features

### Phase 3: Model Development
- **Algorithm Selection**: Compare multiple ML algorithms
- **Hyperparameter Tuning**: Optimize model parameters
- **Cross-Validation**: Ensure model generalization
- **Model Evaluation**: Comprehensive performance metrics
- **Model Persistence**: Save best performing model

### Phase 4: Deployment
- **Web Application**: User-friendly interface for predictions
- **Model Integration**: Real-time prediction capabilities
- **Results Visualization**: Interactive dashboards

## 📊 Model Performance

### Key Metrics Achieved:
- **Accuracy**: 85%+ on test data
- **Precision**: High precision for churn prediction
- **Recall**: Effective identification of at-risk customers
- **F1-Score**: Balanced performance metric
- **ROC-AUC**: Strong discriminative ability

### Business Impact:
- 🎯 **Proactive Retention**: Identify at-risk customers early
- 💰 **Cost Reduction**: Reduce customer acquisition costs
- 📈 **Revenue Protection**: Prevent revenue loss from churn
- 🔄 **Process Optimization**: Data-driven retention strategies

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **Jupyter Notebook**: Interactive development
- **Flask/Streamlit**: Web application framework
- **Pickle**: Model serialization

## 📈 Key Insights

### Customer Behavior Patterns:
- High churn correlation with contract type and tenure
- Payment method significantly impacts retention
- Service usage patterns predict churn likelihood
- Customer demographics influence churn probability

### Actionable Recommendations:
- 🎯 **Target High-Risk Segments**: Focus retention efforts on identified patterns
- 💳 **Payment Optimization**: Encourage automatic payment methods
- 📞 **Proactive Outreach**: Contact customers showing early warning signs
- 🎁 **Personalized Offers**: Tailor retention offers based on customer profile

## 🎯 Applications

This project demonstrates expertise in:
- 📊 **Business Analytics**: Customer behavior analysis
- 🤖 **Machine Learning**: Predictive modeling for business outcomes
- 📈 **Data Science**: End-to-end analytical solution
- 💼 **Business Intelligence**: Actionable insights for decision making


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**[Your Name]**
- GitHub: [Parth Medatwal](https://github.com/ParthMedatwal)
- LinkedIn: [Parth Medatwal](https://www.linkedin.com/in/parth-medatwal-36943220a)
- Email: pmedatwal226@gmail.com

## 🙏 Acknowledgments

- Business analytics community for domain insights
- Open-source data science libraries
- Customer analytics research papers and methodologies

---

⭐ **Star this repository if you found it helpful!**

📧 **Contact me for collaboration opportunities or questions about the analysis!**
