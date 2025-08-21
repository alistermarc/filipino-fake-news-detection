# FilipinoFactCheck: An Analysis of Machine Learning Classifiers for Filipino Fake News Detection

**Authors:** Alister Marc Domilies, Jemar E. Laag

## Abstract

In today's era, misinformation has become widespread despite the scarcity of manual fact-checking resources. While numerous studies focus on creating fake news detection models, most are tailored exclusively for the English language. This study endeavors to fill this void by developing a fake news detection model explicitly designed for the Filipino language. This research adopts an alternative approach to complex deep learning models by utilizing readily available classifiers from the scikit-learn library. By employing the TF-IDF technique to extract essential features, the study evaluates individual, ensemble, and neural network models. The results show that the Stochastic Gradient Descent (SGD) classifier emerged as the overall best-performing model, exhibiting the highest accuracy (96.26%) and the shortest training time. The researchers successfully deployed this model, marking a significant step towards bridging the gap between research and practical application in combating fake news in the Philippines.

## Introduction

Nine out of ten Filipinos acknowledge that fake news is an ongoing issue in the Philippines (Pulse Asia, 2022), and 51% find it difficult to spot (SWS, 2021). This surge in misinformation poses a significant challenge, exacerbated by the vast volume of content on social media. While manual fact-checking organizations exist, their effectiveness is limited. This project aims to develop a simplified yet effective machine learning model capable of determining the authenticity of news articles in the Filipino language.

## Dataset

The project utilizes two main files for its data:

*   `full.csv`: This file contains the dataset of 3,206 Tagalog news articles, sourced from the "Fake News Filipino" benchmark dataset by Cruz et al. (2020). It has an even 50/50 split between real and fake news.
*   `tagalog_stop_words.txt`: A text file containing a list of 147 common Tagalog stop words that are removed during the text preprocessing phase.

## Methodology

The project is structured into two main Python scripts: `train_models.py` for the machine learning pipeline and `app.py` for the web application.

1.  **Data Loading and Splitting**: The `full.csv` dataset is loaded and split into an 80% training set and a 20% testing set.
2.  **Text Preprocessing**: A text processing pipeline is established, which includes converting text to lowercase and removing stop words.
3.  **Feature Extraction**: `TfidfVectorizer` is used to convert the text data into numerical features.
4.  **Model Training and Evaluation**: A suite of ten different classifiers from scikit-learn were trained and evaluated using `train_models.py`.
5.  **Hyperparameter Tuning**: Two experiments were conducted. The first used default hyperparameters. The second used `GridSearchCV` to perform an exhaustive search to find the optimal combination of features and hyperparameters for each classifier.
6.  **Model Deployment**: The best performing model is served using a Flask web application in `app.py`.

## Results

The Stochastic Gradient Descent (SGD) classifier emerged as the top-performing model, boasting the highest test accuracy of **96.26%**. It also had the shortest training time. Four classifiers achieved accuracies exceeding 95%, demonstrating the effectiveness of traditional machine learning models on this task.

### Performance of Tuned Models on Test Data

| Classifier | Accuracy | Precision | Recall | F1-Score | Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stochastic Gradient Descent** | **0.9626** | **0.9764** | **0.9446** | **0.9603** | **0.9791** |
| Multilayer Perceptron (MLP) | 0.9611 | 0.9548 | 0.9642 | 0.9595 | 0.9582 |
| Support Vector Machine (SVC) | 0.9611 | 0.9732 | 0.9446 | 0.9587 | 0.9761 |
| Logistic Regression | 0.9595 | 0.9828 | 0.9316 | 0.9565 | 0.9851 |
| Random Forest | 0.9486 | 0.9628 | 0.9283 | 0.9453 | 0.9672 |
| Gradient Boosting | 0.9377 | 0.9588 | 0.9088 | 0.9331 | 0.9642 |
| AdaBoost | 0.9361 | 0.9493 | 0.9153 | 0.9320 | 0.9552 |
| Multinomial Naive Bayes | 0.9174 | 0.8920 | 0.9414 | 0.9160 | 0.8955 |
| Decision Tree | 0.9081 | 0.9106 | 0.8958 | 0.9031 | 0.9194 |
| K-Nearest Neighbors | 0.8723 | 0.8223 | 0.9349 | 0.8750 | 0.8149 |

## Model Deployment

The best-performing model (SGD) was deployed using a simple Flask web application. This allows users to input text and receive an instant prediction of whether the news is real or fake, demonstrating a practical application of the research.

## Conclusion

This study successfully developed a simple yet effective model for identifying fake news in the Filipino language, achieving an accuracy (96.26%) comparable to more complex deep learning approaches. The Stochastic Gradient Descent classifier proved to be the optimal model due to its high accuracy and efficiency. This work marks a significant advancement by creating an easily deployable tool to combat misinformation in the Philippines.

## Recommendations

To build upon this work, the following steps are recommended:
1.  **Continuous Dataset Updates**: Regularly update the dataset with new articles to keep the model adaptive to evolving language and topics.
2.  **Explore Social Media Data**: Incorporate datasets from social media platforms like Facebook and YouTube, where misinformation is rampant.
3.  **Test More Classifiers**: Include other classifiers in the comparative analysis.
4.  **Broader Hyperparameter Tuning**: Expand the range of hyperparameters to further refine model performance.

## Project Structure

The repository is organized as follows:

-   `app.py`: The Flask web application for model deployment.
-   `train_models.py`: The script for data loading, preprocessing, model training, and evaluation.
-   `requirements.txt`: A list of Python packages required to run the project.
-   `tagalog_stop_words.txt`: The list of Tagalog stop words.
-   `Dataset/full.csv`: The dataset used for training and testing.
-   `Models/`: Contains the trained models (`.joblib` files).
-   `Results/`: Contains CSV files with the detailed performance metrics of the models.
-   `Paper.pdf`: The research paper detailing the study.
-   `Presentation.pdf`: The presentation summarizing the project.
-   `README.md`: This file.
-   `templates/`: Contains the HTML template for the web application.

## How to Use

1.  Ensure you have Python installed.
2.  Install the necessary libraries by running `pip install -r requirements.txt`.
3.  To train the models with default hyperparameters, run `python train_models.py`. The models will be saved in the `Models/No_Hyperparameter_Tuning` directory.
4.  To train the models with hyperparameter tuning, run `python train_models.py --tuned`. The models will be saved in the `Models/With_Hyperparameter_Tuning` directory. This process may take a significant amount of time.
5.  To run the web application, execute `python app.py` and navigate to the provided URL in your web browser.
