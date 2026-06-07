import mne
import numpy as np
from mne.datasets.sleep_physionet import age
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_and_preprocess_data():
    print("--- Pobieranie i ładowanie danych ---")
    subjects = [0]
    fetch_records = age.fetch_data(subjects=subjects, recording=[1])
    
    psg_path = fetch_records[0][0]
    ann_path = fetch_records[0][1]
    
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    annotations = mne.read_annotations(ann_path)
    raw.set_annotations(annotations, emit_warning=False)
    
    raw.filter(0.5, 30.0, picks=['EEG Fpz-Cz', 'EEG Pz-Oz'])
    
    annotation_desc_mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4
    }
    
    events, event_id = mne.events_from_annotations(
        raw, chunk_duration=30.0, event_id=annotation_desc_mapping
    )
    
    epochs = mne.Epochs(
        raw, events=events, event_id=event_id,
        tmin=0., tmax=30.0, baseline=None, preload=True
    )
    
    return epochs

def extract_features(epochs):
    print("--- Ekstrakcja cech PSD ---")
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Sigma': (12, 16),
        'Beta': (16, 30)
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

def train_pipeline():
    epochs = load_and_preprocess_data()
    X, y = extract_features(epochs)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("--- Trening Random Forest ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    target_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    print("\n=== RAPORT KLASYFIKACJI ===")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    accuracy = clf.score(X_test, y_test)
    print(f"\nDokładność: {accuracy:.3f} ({accuracy*100:.1f}%)")

if __name__ == "__main__":
    train_pipeline()
