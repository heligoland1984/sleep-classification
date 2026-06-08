"""
Zaawansowane wizualizacje dla klasyfikacji stadiów snu
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
import pandas as pd

# Ustawienia dla ładniejszych wykresów
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_class_balance(y_train, y_test, target_names):
    """Wykres 1: Balans klas (przed i po SMOTE)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Przed SMOTE
    train_counts = np.bincount(y_train)
    axes[0].bar(target_names, train_counts, color='skyblue', edgecolor='black')
    axes[0].set_title('Zbiór treningowy (przed SMOTE)', fontsize=12)
    axes[0].set_ylabel('Liczba próbek')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Testowy
    test_counts = np.bincount(y_test)
    axes[1].bar(target_names, test_counts, color='lightgreen', edgecolor='black')
    axes[1].set_title('Zbiór testowy', fontsize=12)
    axes[1].set_ylabel('Liczba próbek')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Procentowy rozkład
    train_percent = train_counts / len(y_train) * 100
    axes[2].bar(target_names, train_percent, color='coral', edgecolor='black')
    axes[2].set_title('Rozkład procentowy (treningowy)', fontsize=12)
    axes[2].set_ylabel('Procent (%)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_normalized(y_test, y_pred, target_names):
    """Wykres 2: Znormalizowana macierz pomyłek (procenty)"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Surowa
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, ax=ax1)
    ax1.set_title('Macierz pomyłek - liczba przypadków', fontsize=14)
    ax1.set_ylabel('Rzeczywista klasa')
    ax1.set_xlabel('Przewidziana klasa')
    
    # Znormalizowana
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=target_names, yticklabels=target_names, ax=ax2, 
                vmin=0, vmax=1, center=0.5)
    ax2.set_title('Macierz pomyłek - proporcje (%)', fontsize=14)
    ax2.set_ylabel('Rzeczywista klasa')
    ax2.set_xlabel('Przewidziana klasa')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance_detailed(clf, feature_names, X, y):
    """Wykres 3: Ważność cech z podziałem na kanały"""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top 15 cech
    top_n = 15
    ax1.barh(range(top_n), importances[indices[:top_n]][::-1], color='steelblue')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([feature_names[i] for i in indices[:top_n]][::-1])
    ax1.set_xlabel('Ważność')
    ax1.set_title(f'Top {top_n} najważniejszych cech')
    
    # Ważność dla kanałów
    channel_names = ['Fpz-Cz', 'Pz-Oz']
    channel_importance = {}
    for ch in channel_names:
        ch_indices = [i for i, name in enumerate(feature_names) if ch in name]
        channel_importance[ch] = importances[ch_indices].sum()
    
    ax2.bar(channel_importance.keys(), channel_importance.values(), 
            color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    ax2.set_title('Ważność cech dla kanałów EEG')
    ax2.set_ylabel('Suma ważności')
    
    plt.tight_layout()
    plt.show()

def plot_power_spectrum_by_stage(epochs, y, target_names):
    """Wykres 4: Spektrum mocy dla różnych stadiów snu"""
    from scipy import stats
    
    # Oblicz PSD dla wszystkich epok
    spectrum = epochs.compute_psd(method='welch', fmin=0.5, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    # Średnie spektrum dla każdego stadium
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for stage in range(5):
        stage_mask = y == stage
        if np.sum(stage_mask) > 0:
            stage_psd = np.mean(psds[stage_mask], axis=0)
            
            # Średnia dla obu kanałów
            mean_psd = np.mean(stage_psd, axis=0)
            std_psd = np.std(stage_psd, axis=0)
            
            axes[stage].plot(freqs, mean_psd, color=colors[stage], linewidth=2)
            axes[stage].fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd, 
                                      alpha=0.3, color=colors[stage])
            axes[stage].set_title(f'{target_names[stage]}', fontsize=12)
            axes[stage].set_xlabel('Częstotliwość (Hz)')
            axes[stage].set_ylabel('Moc (dB)')
            axes[stage].grid(True, alpha=0.3)
            axes[stage].set_xlim([0.5, 30])
    
    # Ukryj pusty subplot
    axes[5].set_visible(False)
    
    plt.suptitle('Spektrum mocy EEG dla różnych stadiów snu', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_band_power_comparison(y, features, target_names):
    """Wykres 5: Porównanie mocy w pasmach dla różnych stadiów"""
    # Zakładając, że features zawierają moce dla pasm
    bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for stage in range(5):
        stage_mask = y == stage
        if np.sum(stage_mask) > 0:
            # Wybierz cechy dla tego stadium (zakładając prostą strukturę)
            # To jest uproszczenie - dostosuj do swojej struktury cech
            stage_features = features[stage_mask]
            
            # Dla każdego pasma (pierwsze 5 cech to moce pasm dla kanału 1)
            band_powers = []
            for i, band in enumerate(bands):
                band_power = stage_features[:, i]  # pierwszy kanał
                band_powers.append(band_power)
            
            # Boxplot
            bp = axes[stage].boxplot(band_powers, labels=bands, patch_artist=True)
            for patch, color in zip(bp['boxes'], plt.cm.Set3(range(len(bands)))):
                patch.set_facecolor(color)
            
            axes[stage].set_title(f'{target_names[stage]}', fontsize=12)
            axes[stage].set_ylabel('Moc')
            axes[stage].tick_params(axis='x', rotation=45)
            axes[stage].grid(True, alpha=0.3)
    
    axes[5].set_visible(False)
    plt.suptitle('Moc w pasmach częstotliwości dla różnych stadiów snu', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_classification_metrics_by_class(precision, recall, f1, target_names):
    """Wykres 6: Porównanie metryk dla każdej klasy"""
    x = np.arange(len(target_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, precision, width, label='Precyzja', color='#FF6B6B')
    ax.bar(x, recall, width, label='Recall', color='#4ECDC4')
    ax.bar(x + width, f1, width, label='F1-score', color='#45B7D1')
    
    ax.set_xlabel('Stadium snu', fontsize=12)
    ax.set_ylabel('Wynik', fontsize=12)
    ax.set_title('Metryki klasyfikacji dla poszczególnych stadiów snu', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve(train_sizes, train_scores, val_scores):
    """Wykres 7: Krzywa uczenia się modelu"""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_sizes, train_mean, 'o-', color='#FF6B6B', label='Treningowy')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color='#FF6B6B')
    
    ax.plot(train_sizes, val_mean, 'o-', color='#4ECDC4', label='Walidacyjny')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                    alpha=0.1, color='#4ECDC4')
    
    ax.set_xlabel('Liczba próbek treningowych', fontsize=12)
    ax.set_ylabel('Dokładność', fontsize=12)
    ax.set_title('Krzywa uczenia się modelu', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_error_analysis(y_test, y_pred, X_test, feature_names):
    """Wykres 8: Analiza błędów - które próbki są źle klasyfikowane"""
    errors = y_test != y_pred
    correct = ~errors
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gdzie są błędy?
    axes[0].pie([np.sum(correct), np.sum(errors)], 
                labels=['Poprawne', 'Błędne'], 
                autopct='%1.1f%%',
                colors=['#4ECDC4', '#FF6B6B'],
                explode=(0, 0.1))
    axes[0].set_title('Proporcja błędnych klasyfikacji', fontsize=14)
    
    # Dla jakich klas są błędy?
    unique_classes = np.unique(y_test)
    error_by_class = []
    for cls in unique_classes:
        cls_mask = y_test == cls
        cls_errors = np.sum(errors[cls_mask])
        error_by_class.append(cls_errors / np.sum(cls_mask) if np.sum(cls_mask) > 0 else 0)
    
    axes[1].bar(range(len(unique_classes)), error_by_class, 
                color='#FF6B6B', edgecolor='black')
    axes[1].set_xticks(range(len(unique_classes)))
    axes[1].set_xticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
    axes[1].set_ylabel('Współczynnik błędów')
    axes[1].set_title('Współczynnik błędów dla każdej klasy', fontsize=14)
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_all_visualizations(model, X_test, y_test, y_pred, epochs, y, feature_names, target_names):
    """
    Generuje wszystkie powyższe wykresy
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    print("Generowanie wszystkich wykresów...")
    
    # 1. Balans klas
    print("1/8: Balans klas...")
    # (potrzebujesz y_train - to uproszczone)
    
    # 2. Macierz pomyłek
    print("2/8: Macierz pomyłek...")
    plot_confusion_matrix_normalized(y_test, y_pred, target_names)
    
    # 3. Ważność cech
    print("3/8: Ważność cech...")
    plot_feature_importance_detailed(model, feature_names[:X_test.shape[1]], X_test, y_test)
    
    # 4. Spektrum mocy
    print("4/8: Spektrum mocy...")
    plot_power_spectrum_by_stage(epochs, y, target_names)
    
    # 5. Metryki
    print("5/8: Metryki...")
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    plot_classification_metrics_by_class(precision, recall, f1, target_names)
    
    # 6. Analiza błędów
    print("6/8: Analiza błędów...")
    plot_error_analysis(y_test, y_pred, X_test, feature_names)
    
    print("✓ Wszystkie wykresy wygenerowane!")

