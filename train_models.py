import pandas as pd
import time
import argparse
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Import classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def load_data(data_path, stop_words_path):
    """Loads the dataset and stop words."""
    print("Loading data...")
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None, None, None, None, None

    try:
        with open(stop_words_path, 'r', encoding='utf-8') as file:
            stop_words = [line.strip() for line in file]
    except FileNotFoundError:
        print(f"Error: Stop words file not found at {stop_words_path}")
        stop_words = []

    X_train, X_test, y_train, y_test = train_test_split(
        df['article'], df['label'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, stop_words

def get_classifiers(random_state=42):
    """Returns a list of classifiers to be evaluated."""
    return [
        MultinomialNB(),
        LogisticRegression(random_state=random_state),
        RandomForestClassifier(random_state=random_state),
        DecisionTreeClassifier(random_state=random_state),
        GradientBoostingClassifier(random_state=random_state),
        AdaBoostClassifier(random_state=random_state),
        SVC(random_state=random_state),
        KNeighborsClassifier(),
        SGDClassifier(random_state=random_state),
        MLPClassifier(random_state=random_state, max_iter=1000) # Increased max_iter for convergence
    ]

def get_param_grids(tuned=False):
    """Returns parameter grids for GridSearchCV."""
    if not tuned:
        return {
            'tfidf__max_df': [0.9],
        }

    # Grids for extensive hyperparameter tuning
    return {
        'MultinomialNB': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'classifier__alpha': [0.1, 0.5, 1.0]
        },
        'LogisticRegression': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'classifier__C': [0.1, 1.0, 10.0]
        },
        'RandomForestClassifier': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'classifier__n_estimators': [50, 100, 200]
        },
        'DecisionTreeClassifier': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        },
        'GradientBoostingClassifier': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        },
        'AdaBoostClassifier': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 1.0]
        },
        'SVC': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf']
        },
        'KNeighborsClassifier': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance']
        },
        'SGDClassifier': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'classifier__alpha': [0.0001, 0.001],
            'classifier__penalty': ['l2', 'l1', 'elasticnet']
        },
        'MLPClassifier': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'classifier__alpha': [0.0001, 0.001],
            'classifier__learning_rate_init': [0.001, 0.1]
        }
    }

def run_experiment(X_train, y_train, X_test, y_test, stop_words, classifiers, all_param_grids, output_dir, tuned):
    """Runs the training and evaluation experiment."""
    test_results_list = []
    train_accuracy_list = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for classifier in classifiers:
        classifier_name = type(classifier).__name__
        print(f"----- Training {classifier_name} -----")

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words, lowercase=True, token_pattern=r'\b\w+\b')),
            ('classifier', classifier)
        ])

        if tuned:
            param_grid = all_param_grids.get(classifier_name, {})
        else:
            param_grid = all_param_grids

        grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        total_training_time = time.time() - start_time

        best_model_pipeline = grid_search.best_estimator_
        model_filename = output_dir / f"{classifier_name}_best_model.joblib"
        dump(best_model_pipeline, model_filename)
        print(f"Saved best model to {model_filename}")

        # Save the fitted vectorizer for the best SGD model for our Flask app
        if isinstance(classifier, SGDClassifier):
             vectorizer_filename = 'tfidf_vectorizer.joblib'
             dump(best_model_pipeline.named_steps['tfidf'], vectorizer_filename)
             print(f"Saved TF-IDF vectorizer to {vectorizer_filename} for deployment.")

        # Evaluate on test set
        predictions = best_model_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        # Store results
        test_results_list.append({
            'Classifier': classifier_name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Best Params': grid_search.best_params_
        })
        train_accuracy_list.append({
            'Classifier': classifier_name,
            'Train Accuracy (CV)': grid_search.best_score_,
            'Total Training Time (s)': total_training_time
        })

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Best CV Training Accuracy: {grid_search.best_score_:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")

    return test_results_list, train_accuracy_list

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Train and evaluate fake news detection models.")
    parser.add_argument("--tuned", action="store_true", help="Run with hyperparameter tuning.")
    parser.add_argument("--data-path", type=str, default="full.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--stop-words-path", type=str, default="tagalog_stop_words.txt", help="Path to the stop words text file.")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, stop_words = load_data(args.data_path, args.stop_words_path)

    if X_train is None:
        return

    classifiers = get_classifiers()
    all_param_grids = get_param_grids(tuned=args.tuned)

    if args.tuned:
        print("Running experiment with HYPERPARAMETER TUNING.")
        output_dir = Path("Models") / "With_Hyperparameter_Tuning"
    else:
        print("Running experiment with DEFAULT hyperparameters.")
        output_dir = Path("Models") / "No_Hyperparameter_Tuning"

    test_results, train_results = run_experiment(
        X_train, y_train, X_test, y_test, stop_words, classifiers, all_param_grids, output_dir, args.tuned
    )

    pd.DataFrame(test_results).to_csv(output_dir / "test_results.csv", index=False)
    pd.DataFrame(train_results).to_csv(output_dir / "train_results.csv", index=False)
    print(f"\nResults saved in {output_dir}")

if __name__ == "__main__":
    main()
