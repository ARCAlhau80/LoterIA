# LoterIA - Copilot Instructions

## Project Overview
LoterIA is a lottery prediction system using deep learning with TensorFlow. This Python application analyzes historical lottery data to generate predictions for LOTOFACIL games.

## Key Technologies
- **Python 3.8+** - Main programming language
- **TensorFlow 2.13+** - Deep learning framework for neural network models
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **pyodbc/sqlite3** - Database connectivity
- **scikit-learn** - Machine learning utilities
- **matplotlib** - Data visualization

## Architecture
- **LoterIAConfig**: Global configuration management
- **DatabaseManager**: Database abstraction layer (SQLite/SQL Server)
- **DataProcessor**: Data loading, cleaning, and preprocessing
- **LoterIAModel**: TensorFlow neural network implementation
- **LoterIAPredictor**: Main prediction pipeline

## Development Guidelines
1. Use type hints for all function parameters and return values
2. Follow PEP 8 style guidelines
3. Include comprehensive error handling with try-catch blocks
4. Log important operations using the logging module
5. Use pandas for data manipulation operations
6. Normalize data before feeding to neural networks
7. Save predictions to CSV files with timestamps

## File Structure
```
LoterIA/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── data/                   # Historical lottery data
├── models/                 # Trained TensorFlow models
├── predictions/            # Generated predictions
└── logs/                   # Application logs
```

## Common Tasks
- **Data Import**: Use pandas to read CSV/Excel files
- **Database Operations**: Use DatabaseManager class methods
- **Model Training**: Call LoterIAModel.train() with processed data
- **Predictions**: Use LoterIAPredictor.predict() for new predictions
- **Visualization**: Use matplotlib for data analysis plots

## Best Practices
- Always validate input data before processing
- Use batch processing for large datasets
- Implement proper database connection management
- Save model checkpoints during training
- Include data validation and preprocessing steps
