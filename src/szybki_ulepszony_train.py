import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mne.datasets.sleep_physionet import age
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Ustawienia
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
mne.set_log_level('WARNING')

def load_and_preprocess_data(subjects=list(range(2)), recording=[1]):
    """Ładowanie danych dla 2 pacjentów"""
    print(f"--- Pobieranie danych dla {len(subjects)} pacjentów ---")
    
    all_epochs = []
    for subject in subjects:
        try:
            print(f"Przetwarzanie pacjenta {subject}...")
            fetch_records = age.fetch_data(subjects=[subject], recording=recording)
            
            psg_path = fetch_records[0][0]
            ann_path = fetch_records[0][1]
            
            raw = mne.io.read_raw_edf(psg_path, preload=True)
            annotations = mne.read_annotations(ann_path)
            raw.set_annotations(annotations, emit_warning=False)
            
            raw.filter(0.3, 35.0, picks=['EEG Fpz-Cz', 'EEG Pz-Oz'])
            
            annotation_desc_mapping = {
                'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
                'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4
            }
            
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
    
    if len(all_epochs) > 1:
        combined_epochs = mne.concatenate_epochs(all_epochs)
    else:
        combined_epochs = all_epochs[0]
    
    print(f"\nŁącznie: {len(combined_epochs)} epok")
    return combined_epochs

def extract_features_fast(epochs):
    """Ekstrakcja cech PSD"""
    print("--- Ekstrakcja cech ---")
    
    bands = {
        'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12),
        'Sigma': (12, 16), 'Beta': (16, 30)
    }
    
    spectrum = epochs.compute_psd(method='welch', fmin=0.5, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    features = []
    for epoch_psd in psds:
        epoch_features = []
        for channel_psd in epoch_psd:
            for band, (fmin, fmax) in bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = np.mean(channel_psd[freq_mask])
                epoch_features.append(band_power)
        features.append(epoch_features)
    
    return np.array(features), epochs.events[:, 2]

def plot_confusion_matrix_custom(y_test, y_pred, target_names):
    """Wykres 1: Macierz pomyłek"""
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, ax=ax1)
    ax1.set_title('Macierz pomyłek - liczba przypadków', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Rzeczywista klasa', fontsize=12)
    ax1.set_xlabel('Przewidziana klasa', fontsize=12)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=target_names, yticklabels=target_names, ax=ax2,
                vmin=0, vmax=1, center=0.5)
    ax2.set_title('Macierz pomyłek - proporcje (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Rzeczywista klasa', fontsize=12)
    ax2.set_xlabel('Przewidziana klasa', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance_final(clf, n_features=10):
    """Wykres 2: Ważność cech - finalna działająca wersja"""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    # Nazwy cech
    bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    channels = ['Fpz-Cz', 'Pz-Oz']
    all_names = [f'{ch}_{band}' for ch in channels for band in bands]
    
    # Dopasuj do liczby cech
    if len(all_names) > len(importances):
        all_names = all_names[:len(importances)]
    elif len(all_names) < len(importances):
        all_names = all_names + [f'Cecha_{i}' for i in range(len(all_names), len(importances))]
    
    selected_names = [all_names[i] for i in indices]
    selected_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(selected_names)))
    bars = plt.barh(range(len(selected_names)), selected_importances, color=colors)
    
    plt.yticks(range(len(selected_names)), selected_names)
    plt.xlabel('Ważność', fontsize=12)
    plt.title(f'{n_features} najważniejszych cech dla klasyfikacji snu', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, selected_importances)):
        plt.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_classification_metrics_final(y_test, y_pred, target_names):
    """Wykres 3: Metryki klasyfikacji"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    
    x = np.arange(len(target_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precyzja', color='#FF6B6B', edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall (Czułość)', color='#4ECDC4', edgecolor='black')
    bars3 = ax.bar(x + width, f1, width, label='F1-score', color='#45B7D1', edgecolor='black')
    
    ax.set_xlabel('Stadium snu', fontsize=12)
    ax.set_ylabel('Wynik', fontsize=12)
    ax.set_title('Metryki klasyfikacji dla poszczególnych stadiów snu', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve_final(clf, X, y, cv=3):
    """Wykres 4: Krzywa uczenia"""
    print("   Generowanie krzywej uczenia (może chwilę potrwać)...")
    
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            clf, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_mean, 'o-', color='#FF6B6B', label='Treningowy', linewidth=2, markersize=8)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#FF6B6B')
        
        plt.plot(train_sizes, test_mean, 'o-', color='#4ECDC4', label='Walidacyjny', linewidth=2, markersize=8)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='#4ECDC4')
        
        plt.xlabel('Liczba próbek treningowych', fontsize=12)
        plt.ylabel('Dokładność', fontsize=12)
        plt.title('Krzywa uczenia się modelu Random Forest', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"   ⚠️ Krzywa uczenia pominięta: {e}")

def plot_error_analysis_final(y_test, y_pred, target_names):
    """Wykres 5: Analiza błędów"""
    errors = y_test != y_pred
    correct = ~errors
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Proporcja błędów
    axes[0].pie([np.sum(correct), np.sum(errors)], 
                labels=['✓ Poprawne', '✗ Błędne'], 
                autopct='%1.1f%%',
                colors=['#4ECDC4', '#FF6B6B'],
                explode=(0, 0.1),
                shadow=True)
    axes[0].set_title('Proporcja błędnych klasyfikacji', fontsize=14, fontweight='bold')
    
    # Współczynnik błędów dla każdej klasy
    unique_classes = np.unique(y_test)
    error_rates = []
    class_labels = []
    
    for cls in unique_classes:
        cls_mask = y_test == cls
        if np.sum(cls_mask) > 0:
            cls_errors = np.sum(errors[cls_mask])
            error_rate = cls_errors / np.sum(cls_mask)
            error_rates.append(error_rate)
            class_labels.append(target_names[cls])
    
    bars = axes[1].bar(range(len(class_labels)), error_rates, 
                       color='#FF6B6B', edgecolor='black', linewidth=2)
    axes[1].set_xticks(range(len(class_labels)))
    axes[1].set_xticklabels(class_labels)
    axes[1].set_ylabel('Współczynnik błędów', fontsize=12)
    axes[1].set_title('Współczynnik błędów dla każdej klasy', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, error_rates):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.show()

def plot_power_spectrum_final(epochs, y, target_names):
    """Wykres 6: Spektrum mocy"""
    print("   Generowanie spektrum mocy...")
    
    spectrum = epochs.compute_psd(method='welch', fmin=0.5, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for stage in range(5):
        stage_mask = y == stage
        if np.sum(stage_mask) > 0:
            stage_psd = psds[stage_mask]
            mean_psd = np.mean(stage_psd, axis=(0, 1))
            std_psd = np.std(stage_psd, axis=(0, 1))
            
            axes[stage].plot(freqs, mean_psd, color=colors[stage], linewidth=2, label='Średnia')
            axes[stage].fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd, 
                                      alpha=0.3, color=colors[stage], label='±1 STD')
            axes[stage].set_title(f'{target_names[stage]} (n={np.sum(stage_mask)})', fontsize=12, fontweight='bold')
            axes[stage].set_xlabel('Częstotliwość (Hz)', fontsize=10)
            axes[stage].set_ylabel('Moc (dB)', fontsize=10)
            axes[stage].grid(True, alpha=0.3)
            axes[stage].set_xlim([0.5, 30])
            axes[stage].legend(fontsize=8)
    
    axes[5].set_visible(False)
    plt.suptitle('Spektrum mocy EEG dla różnych stadiów snu', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_band_power_heatmap_final(y, features, target_names):
    """Wykres 7: Heatmapa mocy"""
    bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    channels = ['Fpz-Cz', 'Pz-Oz']
    
    n_bands = len(bands)
    n_channels = len(channels)
    n_stages = len(target_names)
    n_features = min(features.shape[1], n_channels * n_bands)
    
    heatmap_data = np.zeros((n_stages, n_features))
    
    for stage in range(n_stages):
        stage_mask = y == stage
        if np.sum(stage_mask) > 0:
            stage_features = features[stage_mask]
            for idx in range(n_features):
                heatmap_data[stage, idx] = np.mean(stage_features[:, idx])
    
    # Normalizacja
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    heatmap_data_norm = scaler.fit_transform(heatmap_data)
    
    # Etykiety
    labels = []
    for ch in channels:
        for band in bands:
            labels.append(f'{ch}\n{band}')
    labels = labels[:n_features]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heatmap_data_norm, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(n_stages))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(target_names, fontsize=11)
    ax.set_title('Moc w pasmach częstotliwości dla różnych stadiów snu', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Znormalizowana moc')
    
    for i in range(n_stages):
        for j in range(n_features):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                          ha="center", va="center", 
                          color="black" if abs(heatmap_data_norm[i,j]) < 1 else "white", 
                          fontsize=8)
    
    plt.tight_layout()
    plt.show()

def train_fast_pipeline_with_plots():
    """Główny pipeline z wszystkimi wykresami"""
    print("="*60)
    print("FINALNA KLASYFIKACJA STADIÓW SNU - PEŁNA ANALIZA")
    print("="*60)
    
    # 1. Ładowanie danych
    print("\n[1/6] Ładowanie danych...")
    epochs = load_and_preprocess_data(subjects=list(range(2)))
    
    # 2. Ekstrakcja cech
    print("\n[2/6] Ekstrakcja cech...")
    X, y = extract_features_fast(epochs)
    print(f"Kształt danych: {X.shape}")
    
    # 3. Podział danych
    print("\n[3/6] Podział danych...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 4. Normalizacja i balansowanie
    print("\n[4/6] Przygotowanie danych...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # 5. Trenowanie
    print("\n[5/6] Trenowanie modelu...")
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=20,
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    clf.fit(X_train_balanced, y_train_balanced)
    
    # 6. Ewaluacja
    print("\n[6/6] Ewaluacja i generowanie wykresów...")
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    
    target_names = ['Wake (czuwanie)', 'N1', 'N2', 'N3 (sen głęboki)', 'REM']
    
    print("\n" + "="*60)
    print("WYNIKI KLASYFIKACJI")
    print("="*60)
    print(f"\n✅ Dokładność: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print("\n=== RAPORT KLASYFIKACJI ===")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Generowanie wykresów
    print("\n" + "="*60)
    print("GENEROWANIE WYKRESÓW")
    print("="*60)
    
    print("\n📊 Wykres 1/7: Macierz pomyłek...")
    plot_confusion_matrix_custom(y_test, y_pred, target_names)
    
    print("\n📊 Wykres 2/7: Ważność cech...")
    plot_feature_importance_final(clf, n_features=10)
    
    print("\n📊 Wykres 3/7: Metryki klasyfikacji...")
    plot_classification_metrics_final(y_test, y_pred, target_names)
    
    print("\n📊 Wykres 4/7: Krzywa uczenia...")
    plot_learning_curve_final(clf, X_train_balanced, y_train_balanced, cv=3)
    
    print("\n📊 Wykres 5/7: Analiza błędów...")
    plot_error_analysis_final(y_test, y_pred, target_names)
    
    print("\n📊 Wykres 6/7: Spektrum mocy...")
    plot_power_spectrum_final(epochs, y, target_names)
    
    print("\n📊 Wykres 7/7: Heatmapa mocy...")
    plot_band_power_heatmap_final(y, X, target_names)
    
    # Podsumowanie
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    print(f"✓ Liczba pacjentów: 2")
    print(f"✓ Łączna liczba epok: {len(y)}")
    print(f"✓ Liczba cech: {X.shape[1]}")
    print(f"✓ Dokładność modelu: {accuracy:.1%}")
    print("\n✓ Wszystkie 7 wykresów zostało wygenerowanych!")
    
    return clf, X_test, y_test, y_pred

if __name__ == "__main__":
    model, X_test, y_test, y_pred = train_fast_pipeline_with_plots()
    print("\n🎉 Pipeline zakończony pomyślnie! 🎉")