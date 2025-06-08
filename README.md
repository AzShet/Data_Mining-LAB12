# Obesity Level Prediction Project

## Description

This project analyzes an obesity dataset to predict different levels of obesity (or weight status) using Naive Bayes classification models. The process includes data loading, preprocessing (handling missing values, outlier removal, categorical feature encoding), model training, and performance evaluation. The primary analysis and workflow are detailed in the Jupyter Notebook.

## Project Structure

-   `data/`: Contains the raw dataset (`obesidad.csv`).
-   `src/`: Contains the source code, including:
    -   `obesity_analysis.ipynb`: The main Jupyter Notebook with the analysis workflow.
    -   `utils.py`: Utility functions for data preprocessing and model training.
-   `tests/`: Contains unit tests for the utility functions (`test_utils.py`).
-   `requirements.txt`: Lists the Python dependencies for this project.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and activate a virtual environment:**
    *   For Unix/macOS:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   For Windows:
        ```bash
        python -m venv .venv
        .venv\Scripts\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### Analysis Notebook

1.  Ensure your virtual environment is activated and dependencies are installed.
2.  Start Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
3.  Navigate to and open `src/obesity_analysis.ipynb`.
4.  Run the cells in the notebook to perform the analysis.

### Unit Tests

1.  Ensure your virtual environment is activated and dependencies (including pytest) are installed.
2.  From the project root directory, run:
    ```bash
    pytest
    ```
    Or for more verbose output:
    ```bash
    pytest -v
    ```

## Key Technologies Used

-   Python 3
-   Polars
-   NumPy
-   Matplotlib
-   Scikit-learn
-   Category Encoders
-   Jupyter Notebooks/Lab
-   Pytest
