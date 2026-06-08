"""
Ulepszony pipeline klasyfikacji stadiów snu
Autor: Poprawiona wersja z balansowaniem klas i zaawansowanymi cechami
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mne.datasets.sleep_physionet import age
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Ustawienia dla lepszej wydajności
mne.set_log_level('WARNING')

def load_and_preprocess_data(subjects=list(range(5)), recording=[1]):
    """
    Ładowanie i wstępne przetwarzanie danych dla wielu pacjentów
    
    Parameters:
    -----------
    subjects : list
        Lista numerów badanych (0-19)
    recording : list
        Lista numerów nagrań (1 lub 2)
    """
    print(f"--- Pobieranie danych dla {len(subjects)} pacjentów ---")
    
    all_epochs = []
    
    for subject in subjects:
        try:
            print(f"Przetwarzanie pacjenta {subject}...")
            fetch_records = age.fetch_data(subjects=[subject], recording=recording)
            
            psg_path = fetch_records[0][0]
            ann_path = fetch_records[0][1]
            
            # Wczytanie sygnału
            raw = mne.io.read_raw_edf(psg_path, preload=True)
            annotations = mne.read_annotations(ann_path)
            raw.set_annotations(annotations, emit_warning=False)
            
            # Filtrowanie - szersze pasmo dla lepszej ekstrakcji cech
            raw.filter(0.3, 35.0, picks=['EEG Fpz-Cz', 'EEG Pz-Oz'])
            
            # Mapowanie stadiów snu
            annotation_desc_mapping = {
                'Sleep stage W': 0,      # Wake
                'Sleep stage 1': 1,      # N1
                'Sleep stage 2': 2,      # N2
                'Sleep stage 3': 3,      # N3
                'Sleep stage 4': 3,      # N3 (łączymy)
                'Sleep stage R': 4       # REM
            }
            
            # Tworzenie epok
            events, event_id = mne.events_from_annotations(
                raw, chunk_duration=30.0, event_id=annotation_desc_mapping
            )
            
            epochs = mne.Epochs(
                raw, events=events, event_id=event_id,
                tmin=0., tmax=30.0, baseline=None, preload=True
            )
            
            all_epochs.append(epochs)
            print(f"  ✓ Pacjent {subject}: {len(epochs)} epok")
            
        except Exception as e:
            print(f"  ✗ Błąd dla pacjenta {subject}: {e}")
            continue
    
    # Połączenie epok od wszystkich pacjentów
    if len(all_epochs) > 1:
        combined_epochs = mne.concatenate_epochs(all_epochs)
    else:
        combined_epochs = all_epochs[0]
    
    print(f"\nŁącznie: {len(combined_epochs)} epok")
    return combined_epochs

def extract_advanced_features(epochs):
    """
    Ekstrakcja zaawansowanych cech z sygnałów EEG
    
    Cechy:
    1. Moc w pasmach częstotliwości (PSD)
    2. Entropia widmowa
    3. Stosunki mocy między pasmami
    4. Statystyki sygnału
    """
    print("--- Ekstrakcja zaawansowanych cech ---")
    
    # Definicje pasm częstotliwości
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Sigma': (12, 16),
        'Beta': (16, 30),
        'Gamma': (30, 45)  # Dodane pasmo gamma
    }
    
    # Obliczenie PSD z lepszą rozdzielczością
    spectrum = epochs.compute_psd(method='welch', fmin=0.5, fmax=45.0, 
                                  n_fft=2048, n_overlap=1024)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    features = []
    labels = epochs.events[:, 2]
    
    # Dla każdej epoki
    for epoch_idx, epoch_psd in enumerate(psds):
        epoch_features = []
        
        # Dla każdego kanału
        for channel_idx, channel_psd in enumerate(epoch_psd):
            
            # 1. Moc w pasmach częstotliwości
            band_powers = {}
            for band_name, (fmin, fmax) in bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = np.mean(channel_psd[freq_mask])
                band_powers[band_name] = band_power
                epoch_features.append(band_power)
            
            # 2. Entropia widmowa (miara złożoności sygnału)
            power_spectrum = channel_psd
            power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
            spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-12))
            epoch_features.append(spectral_entropy)
            
            # 3. Stosunki mocy (ważne dla rozróżniania stadiów)
            if 'Theta' in band_powers and 'Beta' in band_powers:
                theta_beta_ratio = band_powers['Theta'] / (band_powers['Beta'] + 1e-8)
                epoch_features.append(theta_beta_ratio)
            
            if 'Alpha' in band_powers and 'Beta' in band_powers:
                alpha_beta_ratio = band_powers['Alpha'] / (band_powers['Beta'] + 1e-8)
                epoch_features.append(alpha_beta_ratio)
            
            if 'Delta' in band_powers and 'Theta' in band_powers:
                delta_theta_ratio = band_powers['Delta'] / (band_powers['Theta'] + 1e-8)
                epoch_features.append(delta_theta_ratio)
            
            # 4. Całkowita moc sygnału
            total_power = np.sum(channel_psd)
            epoch_features.append(total_power)
            
            # 5. Częstotliwość szczytowa
            peak_freq_idx = np.argmax(channel_psd)
            peak_frequency = freqs[peak_freq_idx]
            epoch_features.append(peak_frequency)
        
        features.append(epoch_features)
    
    return np.array(features), labels

def get_optimal_model():
    """
    Znajdowanie optymalnych parametrów dla Random Forest
    """
    # Definicja siatki parametrów do przeszukania
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Pipeline z normalizacją, SMOTE i klasyfikatorem
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Grid search z walidacją krzyżową
    grid_search = GridSearchCV(
        pipeline, param_grid, 
        cv=5, scoring='f1_weighted', 
        n_jobs=-1, verbose=1
    )
    
    return grid_search

def plot_confusion_matrix_custom(y_test, y_pred, target_names):
    """
    Zaawansowana wizualizacja macierzy pomyłek
    """
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Macierz surowa
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, ax=ax1)
    ax1.set_title('Macierz pomyłek - liczba przypadków', fontsize=14)
    ax1.set_ylabel('Rzeczywista klasa', fontsize=12)
    ax1.set_xlabel('Przewidziana klasa', fontsize=12)
    
    # Macierz znormalizowana
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=target_names, yticklabels=target_names, ax=ax2)
    ax2.set_title('Macierz pomyłek - proporcje', fontsize=14)
    ax2.set_ylabel('Rzeczywista klasa', fontsize=12)
    ax2.set_xlabel('Przewidziana klasa', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return cm

def plot_feature_importance(clf, feature_names, top_n=20):
    """
    Wizualizacja ważności cech
    """
    # Pobierz ważność cech z ostatniego klasyfikatora w pipeline
    if hasattr(clf, 'named_steps'):
        classifier = clf.named_steps['classifier']
    else:
        classifier = clf
        
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} najważniejszych cech', fontsize=16)
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Cechy', fontsize=12)
    plt.ylabel('Ważność', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y_train, y_test, target_names):
    """
    Wizualizacja rozkładu klas przed i po SMOTE
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    ax1.bar(target_names, train_counts, color='skyblue', edgecolor='black')
    ax1.set_title('Rozkład klas - zbiór treningowy', fontsize=14)
    ax1.set_ylabel('Liczba próbek', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(target_names, test_counts, color='lightgreen', edgecolor='black')
    ax2.set_title('Rozkład klas - zbiór testowy', fontsize=14)
    ax2.set_ylabel('Liczba próbek', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def save_model_results(y_test, y_pred, accuracy, cv_scores):
    """
    Zapisywanie wyników do pliku
    """
    results = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Zapisz do CSV
    df_results = pd.DataFrame([results])
    df_results.to_csv('sleep_classification_results.csv', index=False)
    print("\n✓ Wyniki zapisane do 'sleep_classification_results.csv'")
    
    # Zapisz szczegółowy raport
    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    print("✓ Raport zapisany do 'classification_report.txt'")

def train_improved_pipeline():
    """
    Główny pipeline trenowania z wszystkimi ulepszeniami
    """
    print("="*60)
    print("ZAAWANSOWANA KLASYFIKACJA STADIÓW SNU")
    print("="*60)
    
    # 1. Załaduj dane (więcej pacjentów)
    print("\n[1/6] Ładowanie danych...")
    epochs = load_and_preprocess_data(subjects=list(range(5)))  # 5 pacjentów
    
    # 2. Ekstrakcja cech
    print("\n[2/6] Ekstrakcja cech...")
    X, y = extract_advanced_features(epochs)
    
    print(f"\nKształt danych: {X.shape}")
    print(f"Liczba klas: {len(np.unique(y))}")
    print(f"Rozkład klas: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # 3. Podział na zbiór treningowy i testowy
    print("\n[3/6] Podział danych...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Wizualizacja rozkładu klas
    target_names = ['Wake (NREM)', 'N1', 'N2', 'N3', 'REM']
    plot_class_distribution(y_train, y_test, target_names)
    
    # 4. Przygotowanie modelu z optymalizacją
    print("\n[4/6] Optymalizacja modelu...")
    grid_search = get_optimal_model()
    
    # 5. Trenowanie z walidacją krzyżową
    print("\n[5/6] Trenowanie modelu...")
    grid_search.fit(X_train, y_train)
    
    # Najlepszy model
    best_model = grid_search.best_estimator_
    print(f"\nNajlepsze parametry: {grid_search.best_params_}")
    
    # Walidacja krzyżowa
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_weighted')
    print(f"Walidacja krzyżowa (F1-weighted): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # 6. Ewaluacja
    print("\n[6/6] Ewaluacja modelu...")
    y_pred = best_model.predict(X_test)
    accuracy = best_model.score(X_test, y_test)
    
    # Wyświetl wyniki
    print("\n" + "="*60)
    print("WYNIKI KLASYFIKACJI")
    print("="*60)
    print(f"\nDokładność: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print("\n=== RAPORT KLASYFIKACJI ===")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Wizualizacje
    cm = plot_confusion_matrix_custom(y_test, y_pred, target_names)
    
    # Ważność cech
    feature_names = []
    for channel in ['Fpz-Cz', 'Pz-Oz']:
        for band in ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']:
            feature_names.append(f'{channel}_{band}_power')
    feature_names.extend(['Spectral_Entropy', 'Theta/Beta_Ratio', 'Alpha/Beta_Ratio', 
                          'Delta/Theta_Ratio', 'Total_Power', 'Peak_Frequency'])
    feature_names = feature_names * 2  # Dla dwóch kanałów
    
    plot_feature_importance(best_model, feature_names[:X.shape[1]], top_n=15)
    
    # Zapisz wyniki
    save_model_results(y_test, y_pred, accuracy, cv_scores)
    
    # Podsumowanie
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    print(f"✓ Liczba pacjentów: 5")
    print(f"✓ Łączna liczba epok: {len(y)}")
    print(f"✓ Liczba cech: {X.shape[1]}")
    print(f"✓ Dokładność: {accuracy:.1%}")
    print(f"✓ F1-weighted (CV): {cv_scores.mean():.3f}")
    
    return best_model, X_test, y_test, y_pred

if __name__ == "__main__":
    # Uruchom ulepszony pipeline
    model, X_test, y_test, y_pred = train_improved_pipeline()
    
    print("\n✓ Pipeline zakończony pomyślnie!")
    print("\nAby użyć modelu do predykcji nowych danych:")
    print("  predictions = model.predict(nowe_dane)")