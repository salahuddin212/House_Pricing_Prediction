# California Housing Price Prediction 🏡

## Executive Summary

This project demonstrates a complete end-to-end machine learning workflow to build a regression model that predicts median housing values in California. The objective was not just to build a model, but to strategically analyze, clean, and prepare real-world data, addressing common challenges like missing values, skewed distributions, and categorical data.

This project showcases proficiency in:

  * **Exploratory Data Analysis (EDA)**
  * **Data Cleaning & Preprocessing**
  * **Problem Identification & Strategic Solutioning**
  * **Feature Engineering (Conceptual)**
  * **Python, Pandas, Scikit-Learn, and Matplotlib**

-----

## Key Challenges & Solutions

During the initial data exploration, several critical issues were identified that would negatively impact model performance if left unaddressed. Here’s a summary of the problems and the strategies devised to solve them:

| Challenge Identified | Description & Impact | Solution Strategy |
| :--- | :--- | :--- |
| **1. Missing Data** | The `total_bedrooms` feature had **207 missing values**. This would cause most Scikit-Learn algorithms to fail. Simply dropping the rows would result in data loss. | The chosen strategy would be **imputation**. The missing values would be filled with the **median** of the column, as the median is more robust to outliers than the mean. |
| **2. Skewed Feature Distributions** | Histograms revealed that features like `total_rooms`, `population`, and `households` were heavily **tail-heavy**. Models perform better on more normally distributed data. | Apply a **logarithmic transformation** (`np.log`) to these skewed features to compress their range and reduce the impact of extreme values, leading to a more normalized distribution. |
| **3. Capped Data Values** | The `housing_median_age` and the target variable `median_house_value` appeared to be **capped at their maximum values**. This creates an artificial ceiling and can mislead the model. | This is a data integrity issue. The solution would be to discuss with stakeholders. If they confirm it's a cap, the model's predictions would also need to be capped at that value, or the capped entries could be removed from the training set to prevent the model from learning this artificial limit. |
| **4. Non-Numerical Categorical Data** | The `ocean_proximity` feature was a categorical text object. Machine learning models require numerical input. | Implement **One-Hot Encoding** to convert the five distinct categories (`<1H OCEAN`, `INLAND`, etc.) into a numerical format that the model can process without assuming an incorrect ordinal relationship. |

-----

## methodological Workflow

The project was structured to mirror a professional data science pipeline.

### 1\. Data Acquisition & Setup

A script was created to automate the process of fetching the dataset, ensuring the project is easily reproducible. The initial setup also includes importing necessary libraries and defining paths for saving figures, establishing a clean and organized workspace.

### 2\. Exploratory Data Analysis (EDA)

A deep dive into the dataset was performed to uncover its underlying structure and identify the challenges mentioned above.

  * `.info()` was used to get a high-level overview of data types and null counts, immediately flagging the missing `total_bedrooms`.
  * `.describe()` provided a statistical summary, giving the first hints of potential outliers and scaling issues (e.g., large standard deviations in `total_rooms`).
  * `.hist()` was crucial for visualizing the distributions of all numerical features, which clearly exposed the skewed data and capped values.
  * `.value_counts()` was used on the `ocean_proximity` feature to understand its categorical distribution.

### 3\. Feature Engineering & Preprocessing

Based on the EDA, a preprocessing pipeline was conceptualized to transform the raw data into a clean, model-ready format. This is the most critical step in ensuring model accuracy. The key idea is to create new features that provide stronger signals to the model.

**Planned Feature Creations:**

  * `rooms_per_household`: To normalize the room count by the number of households.
  * `population_per_household`: To get a measure of household size.
  * `bedrooms_per_room`: To create a ratio of bedrooms to total rooms.

-----

## 🚀 Future Work & Next Steps

  * **Implement Preprocessing Pipeline**: Build a full Scikit-Learn pipeline to execute the cleaning and feature engineering steps.
  * **Model Training**: Train several regression models (e.g., Linear Regression, Decision Tree, Random Forest) to establish a performance baseline.
  * **Model Evaluation & Tuning**: Evaluate the models using cross-validation and the **Root Mean Squared Error (RMSE)** metric. Fine-tune the best-performing model's hyperparameters using `GridSearchCV`.
  * **Deployment**: Deploy the final, trained model as a simple API for real-world use.

-----

## 🛠️ Requirements & Usage

To run this project:

1.  **Prerequisites**: Ensure you have Python (\>=3.5) and the following libraries installed:
    ```bash
    pip install numpy pandas scikit-learn matplotlib jupyter
    ```
2.  **Run the Notebook**: Open the `.ipynb` file in a Jupyter environment and execute the cells sequentially.
