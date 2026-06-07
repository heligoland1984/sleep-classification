"""
Skrypt do pobierania danych Sleep PhysioNet
"""
from mne.datasets.sleep_physionet import age

def download_sleep_data(subjects=[0], recording=[1]):
    """
    Pobiera dane z bazy Sleep-EDF
    """
    print(f"Pobieranie danych dla badanych: {subjects}, nagranie: {recording}")
    fetch_records = age.fetch_data(subjects=subjects, recording=recording)
    print(f"Dane pobrane pomyślnie!")
    return fetch_records

if __name__ == "__main__":
    download_sleep_data(subjects=[0], recording=[1])
