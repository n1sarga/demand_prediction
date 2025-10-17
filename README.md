# **Supply Chain Demand Prediction**

## **Acknowledgement**
This is the final project of the CSE407 (Green Computing) course. 
This work has been accepted at the International Conference on Data Mining and Information Security 2025 ([ICDMIS2025](http://icdmis.ikrf.in/)). 
The work was supervised by Prof. [Dr. Ahmed Wasif Reza](https://fse.ewubd.edu/computer-science-engineering/faculty-view/wasif), assisted by K. M. Safin Kamal, and conducted by,
| Name         | Affiliation                                      |
|--------------------|--------------------------------------------------|
| Nisarga Mridha (Me)      | East West University |
| Shah Newaz Parvez Shuvo            | Khulna University of Engineering and Technology        |
| Sumit Kumar Rahut         | East West University           |
| Mehedi Hasan Rabbi         | Southeast University  |
| MD. Rafiul Azam           | Khulna University of Engineering and Technology |
| Jannatul Naeem Tilotama| East West University       |
| Khalid Saifullah Fuad         | Southeast University                           |
| Md. Golam Rabbanie Babu   | Southeast University          |
| Md. Mahadi Hasan           | University of South Asia     |

## **Dataset**
The dataset used for this project can be accessed from [Kaggle](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset).
## **Working Procedure**
<img width="1350" height="1080" alt="Your paragraph text" src="https://github.com/user-attachments/assets/0e5806cd-d723-48bf-a70c-3d8ad2a0bcf5" />

## Results
| Model                     | MSE      | RMSE   | MAE   | R²   |
|----------------------------|----------|--------|-------|------|
| **Linear Regression**          | **75.03**    | **8.66**   | **7.47**  | **0.99** |
| Lasso                      | 141.38   | 11.89  | 9.66  | 0.98 |
| Ridge                      | 75.04    | 8.66   | 7.47  | 0.99 |
| ElasticNet                 | 2465.66  | 49.65  | 39.75 | 0.79 |
| Gradient Boosting Regressor| 112.60   | 10.61  | 8.73  | 0.99 |
| XGBoost                    | 91.77    | 9.58   | 8.02  | 0.99 |
| LightGBM                   | 85.20    | 9.23   | 7.83  | 0.99 |
| Support Vector Regressor   | 349.56   | 18.69  | 13.19 | 0.97 |
| K-Nearest Neighbors        | 2296.39  | 47.92  | 38.63 | 0.80 |
| Multilayer Perceptron      | 78.29    | 8.84   | 7.58  | 0.99 |
| Decision Tree              | 210.98   | 14.52  | 11.68 | 0.98 |
| Random Forest              | 98.70    | 9.93   | 8.28  | 0.99 |

## Best Model Summary
| **Model Summary** |  |  |  |
|:----------------:|---|---|---|
| **Metric** | **Value** |  |  |
| R² | 0.994 |  |  |
| Adjusted R² | 0.994 |  |  |
| F-statistic (p-value) | 5.04 × 10⁵ (p < 0.001) |  |  |
| Number of observations | 73,100 |  |  |
| **Significant Predictors (p < 0.05)** |  |  |  |
| **Predictor** | **Coefficient** | **Std. Error** | **p-value** |
| Sales | 108.86 | 0.040 | < 0.001 |
| Region (South) | 0.091 | 0.039 | 0.020 |
| Region (West) | 0.077 | 0.039 | 0.050 |
| *Inventory* | 0.076 | 0.040 | *0.055 (Marginal)* |

## Packages Used:
(Will Be Updated)
