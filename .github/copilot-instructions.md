# AI Agent Instructions for Machine Learning Project

## Project Structure
This is a machine learning project following a standard research organization pattern:
- `data/`: Contains raw and processed datasets
  - `raw/`: Original, immutable input data
  - `processed/`: Cleaned and transformed data ready for modeling
- `models/`: Trained model artifacts and serialized objects
- `outputs/`: Results and artifacts
  - `logs/`: Training and execution logs
  - `metrics/`: Model performance metrics
  - `visualizations/`: Plots, charts and visual results
- `src/`: Source code modules
- `config.yaml`: Configuration parameters
- `requirements.txt`: Python dependencies
- `main.py`: Entry point for model training/inference

## Key Conventions
1. Data Processing Flow:
   - Raw data goes in `data/raw/`
   - Processed/transformed data saved to `data/processed/`
   - Final model outputs and visualizations go to respective folders under `outputs/`

2. Configuration Management:
   - All configurable parameters should be defined in `config.yaml`
   - No hardcoded parameters in source code

3. Code Organization:
   - Core ML pipeline code lives in `src/`
   - `main.py` orchestrates the complete workflow

## Development Workflow
1. Set up environment:
   ```bash
   python -m venv venv
   pip install -r requirements.txt
   ```

2. Data Processing:
   - Place raw data in `data/raw/`
   - Process using pipeline in `src/`
   - Validate processed outputs in `data/processed/`

3. Model Development:
   - Configure model parameters in `config.yaml`
   - Train via `main.py`
   - Review metrics and visualizations in `outputs/`

## Best Practices
- Keep raw data immutable - never modify files in `data/raw/`
- Log all experiments and metrics to `outputs/logs/`
- Save visualizations with clear naming in `outputs/visualizations/`
- Document model performance metrics in `outputs/metrics/`