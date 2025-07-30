# ☀️ Solar Energy Production Predictor

A comprehensive machine learning application that predicts annual photovoltaic (PV) energy production using advanced XGBoost algorithms and interactive Streamlit interface.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- All required model files (see below)

### Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have all these files in your directory:
   # - app.py
   # - requirements.txt
   # - xgboost_best_model.pkl
   # - scaler.pkl
   # - label_encoders.pkl
   # - Solar Energy.csv
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL shown in your terminal

## 📁 Required Files

Make sure you have all these files in your project directory:

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `xgboost_best_model.pkl` - Trained XGBoost model
- `scaler.pkl` - Data scaler for preprocessing
- `label_encoders.pkl` - Label encoders for categorical variables
- `Solar Energy.csv` - Training dataset
- `README.md` - This file

## 🎯 Features

### 🔮 Prediction Interface
- **Interactive Forms**: Easy-to-use input forms for solar system specifications
- **Real-time Predictions**: Instant energy production estimates
- **Confidence Metrics**: Model confidence indicators and performance metrics
- **Additional Calculations**: Daily, monthly production and efficiency metrics

### 📊 Data Analysis
- **Comprehensive Visualizations**: Energy production distributions, system size analysis
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Statistical Overview**: Dataset statistics and insights

### 📈 Model Performance
- **Performance Metrics**: R² score, RMSE, MAE comparisons
- **Feature Importance**: Detailed feature importance analysis
- **Model Comparison**: Multiple algorithm performance comparison

### 🏠 Home Dashboard
- **Quick Statistics**: Overview of dataset and model performance
- **Sample Data**: Preview of the training dataset
- **Project Overview**: Key highlights and capabilities

## 🎨 User Interface

The app features a modern, responsive design with:

- **Navigation Sidebar**: Easy access to all features
- **Interactive Forms**: User-friendly input interfaces
- **Beautiful Visualizations**: Rich charts and graphs
- **Responsive Layout**: Works on desktop and mobile devices
- **Professional Styling**: Clean, modern design with custom CSS

## 📊 Model Performance

- **R² Score**: 0.865 (86.5% accuracy)
- **RMSE**: 10,387 kWh
- **Training Data**: 206,776 solar energy projects
- **Features**: 9 key predictive variables

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: XGBoost Regressor
- **Preprocessing**: StandardScaler for numerical features
- **Encoding**: Label encoding for categorical variables
- **Feature Engineering**: Target encoding for high-cardinality features

### Key Features Used
1. **System Specifications**: PV system sizes (DC/AC)
2. **Location Data**: County, City/Town
3. **Utility Information**: Utility company
4. **Technical Details**: Metering method, developer
5. **Storage**: Energy storage system size
6. **Project Scale**: Number of projects

### Technology Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **ML Framework**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly

## 🚀 Usage Guide

### Making Predictions

1. **Navigate to "Make Prediction"**
   - Click on the sidebar navigation
   - Select "🔮 Make Prediction"

2. **Enter System Details**
   - **System Specifications**: Enter PV system sizes (DC/AC)
   - **Location**: Select county and city/town
   - **Utility**: Choose utility company
   - **Technical Details**: Select metering method and developer
   - **Storage**: Enter energy storage system size (if applicable)
   - **Project Scale**: Specify number of projects

3. **Submit and View Results**
   - Click "🚀 Predict Energy Production"
   - View predicted annual energy production
   - Check additional metrics (daily/monthly production, efficiency)
   - Review model confidence indicators

### Exploring Data Analysis

1. **Navigate to "Data Analysis"**
   - View comprehensive dataset statistics
   - Explore energy production distributions
   - Analyze system size vs energy production relationships
   - Compare utilities by average production

### Understanding Model Performance

1. **Navigate to "Model Performance"**
   - Review performance metrics (R², RMSE, MAE)
   - Explore feature importance rankings
   - Compare different model algorithms
   - Understand model capabilities

## 🔍 Troubleshooting

### Common Issues

1. **"Error loading model files"**
   - Ensure all `.pkl` files are in the same directory as `app.py`
   - Check file permissions and paths

2. **"Module not found" errors**
   - Install dependencies: `pip install -r requirements.txt`
   - Ensure Python environment is activated

3. **"Data loading errors"**
   - Verify `Solar Energy.csv` is in the project directory
   - Check file format and encoding

4. **Performance issues**
   - Close other applications to free up memory
   - Consider using a smaller sample of data for analysis

### File Structure
```
project_directory/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── xgboost_best_model.pkl   # Trained XGBoost model
├── scaler.pkl               # Data scaler
├── label_encoders.pkl       # Label encoders
├── Solar Energy.csv         # Training dataset
└── README.md               # This file
```

## 📞 Support

For questions, issues, or feature requests:
- Check the troubleshooting section above
- Review the model performance metrics
- Ensure all required files are present

## 📈 Future Enhancements

Potential improvements for future versions:
- **Real-time Data Integration**: Connect to live solar energy APIs
- **Advanced Analytics**: More sophisticated data analysis tools
- **Export Functionality**: Download predictions and reports
- **User Authentication**: Multi-user support with saved predictions
- **Mobile App**: Native mobile application
- **API Endpoints**: REST API for integration with other systems

---

**Built with ❤️ using Streamlit, XGBoost, and Python** 