import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

def run_classification_experiment(n_iterations=30):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Definindo os modelos
    models = {
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC()
    }

    # Ajustando hiperparâmetros para KNN e SVM
    param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
    param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    
    grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
    grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
    
    best_knn = grid_search_knn.fit(X, y).best_estimator_
    best_svm = grid_search_svm.fit(X, y).best_estimator_

    # Atualizando o dicionário de modelos com os melhores encontrados
    models['KNN'] = best_knn
    models['SVM'] = best_svm

    # Resultados
    results = {model_name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for model_name in models.keys()}

    # Rodando a experimentação
    for i in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculando as métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[model_name]['accuracy'].append(accuracy)
            results[model_name]['precision'].append(precision)
            results[model_name]['recall'].append(recall)
            results[model_name]['f1'].append(f1)
    
    return results

def plot_results(results):
    model_names = list(results.keys())
    
    mean_accuracies = [np.mean(results[model]['accuracy']) for model in model_names]
    std_accuracies = [np.std(results[model]['accuracy']) for model in model_names]
    
    mean_precisions = [np.mean(results[model]['precision']) for model in model_names]
    mean_recalls = [np.mean(results[model]['recall']) for model in model_names]
    mean_f1s = [np.mean(results[model]['f1']) for model in model_names]

    # Gráfico de barras para acurácia
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, mean_accuracies, yerr=std_accuracies, capsize=5, alpha=0.7)
    plt.xlabel('Modelos')
    plt.ylabel('Acurácia')
    plt.title('Comparação de Modelos de Classificação - Acurácia')
    plt.show()

    # Gráfico de barras para as outras métricas
    metrics = ['Precision', 'Recall', 'F1 Score']
    metric_values = [mean_precisions, mean_recalls, mean_f1s]
    
    for metric, values in zip(metrics, metric_values):
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, values, alpha=0.7)
        plt.xlabel('Modelos')
        plt.ylabel(f'{metric}')
        plt.title(f'Comparação de Modelos de Classificação - {metric}')
        plt.show()

if __name__ == '__main__':
    # Executar o experimento
    results = run_classification_experiment()

    # Exibir os resultados
    print("\n--- Resultados da Classificação ---")
    for model_name, metrics in results.items():
        print(f"\nModelo: {model_name}")
        for metric_name, metric_values in metrics.items():
            mean_metric = np.mean(metric_values)
            std_metric = np.std(metric_values) if metric_name == 'accuracy' else '-'
            print(f"  Média da {metric_name.capitalize()}: {mean_metric:.4f}")
            if std_metric != '-':
                print(f"  Desvio Padrão da {metric_name.capitalize()}: {std_metric:.4f}")

    # Visualizar os resultados
    plot_results(results)
