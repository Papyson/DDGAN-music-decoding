import os
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
import librosa.display

# Helper function to load audio and generate spectrogram
def get_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)  # Load audio with default sample rate
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # Specify y= for the first argument
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

# Helper function to load audio and extract mel-cepstral coefficients
def get_mel_cepstral_coefficients(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract the first 13 MFCCs
    return mfcc

# Function to calculate Mel-Cepstral Distortion (MCD)
def calculate_mcd(mfcc1, mfcc2):
    # Ensure both have the same number of frames
    min_length = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_length]
    mfcc2 = mfcc2[:, :min_length]

    # Calculate squared differences
    diff = mfcc1 - mfcc2
    dist = np.sum(diff ** 2, axis=0)

    # Calculate the MCD metric
    mcd = (10 / np.log(10)) * np.sqrt(np.mean(dist))
    return mcd

def analyze_audio_folders(original_folder, gan_folder):
    # Dictionary to store results grouped by song
    results_by_song = {}

    # Process each song in the original folder
    for song_file in os.listdir(original_folder):
        if song_file.endswith(".wav") and os.path.exists(os.path.join(gan_folder, song_file)):
            participant, song_name = song_file.split('_', 1)
            song_name = song_name.replace(".wav", "")

            # Load original and GAN audio
            original_path = os.path.join(original_folder, song_file)
            gan_path = os.path.join(gan_folder, song_file)

            # Generate spectrograms for both original and GAN versions
            original_spectrogram = get_spectrogram(original_path)
            gan_spectrogram = get_spectrogram(gan_path)

            # Flatten spectrograms for correlation calculation
            original_flat = original_spectrogram.flatten()
            gan_flat = gan_spectrogram.flatten()

            # Calculate Pearson correlation
            correlation, _ = pearsonr(original_flat, gan_flat)

            # Extract mel-cepstral coefficients
            original_mfcc = get_mel_cepstral_coefficients(original_path)
            gan_mfcc = get_mel_cepstral_coefficients(gan_path)

            # Calculate MCD
            mcd_score = calculate_mcd(original_mfcc, gan_mfcc)

            # Store results in song's dictionary
            if song_name not in results_by_song:
                results_by_song[song_name] = {
                    'participants': [],
                    'correlations': [],
                    'mcd_scores': [],
                    'spectrogram_pairs': []
                }

            # Add data to the song's results
            results_by_song[song_name]['participants'].append(participant)
            results_by_song[song_name]['correlations'].append(correlation)
            results_by_song[song_name]['mcd_scores'].append(mcd_score)
            results_by_song[song_name]['spectrogram_pairs'].append((participant, original_spectrogram, gan_spectrogram))

            # Display results
            print(f"Song: {song_name} | Participant: {participant} | Pearson Correlation: {correlation:.4f} | MCD Score: {mcd_score:.4f}")

        else:
            print(f"Skipping {song_file}: Corresponding GAN file not found.")

    # Visualization: Per-Song Per-Participant Analysis with Spectrograms
    for song_name, results in results_by_song.items():
        for participant, original_spectrogram, gan_spectrogram in results['spectrogram_pairs']:
            plt.figure(figsize=(14, 6))

            # Original spectrogram
            plt.subplot(1, 2, 1)
            librosa.display.specshow(original_spectrogram, sr=44100, cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Original Spectrogram - {song_name} (Participant: {participant})')

            # GAN-reconstructed spectrogram
            plt.subplot(1, 2, 2)
            librosa.display.specshow(gan_spectrogram, sr=44100, cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'GAN-Reconstructed Spectrogram - {song_name} (Participant: {participant})')

            plt.tight_layout()
            plt.show()

    # Calculate average correlation and MCD scores for each song across participants
    average_correlation_per_song = {}
    average_mcd_per_song = {}

    for song_name, results in results_by_song.items():
        average_correlation_per_song[song_name] = np.mean(results['correlations'])
        average_mcd_per_song[song_name] = np.mean(results['mcd_scores'])

    # Display results in a table
    average_results_df = pd.DataFrame({
        'Song': list(average_correlation_per_song.keys()),
        'Average Correlation': list(average_correlation_per_song.values()),
        'Average MCD Score': list(average_mcd_per_song.values())
    })

    print("\nAverage Correlation and MCD Scores for Each Song Across Participants:")
    print(average_results_df)

    # Visualization: Bar plot for average correlation per song
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Song', y='Average Correlation', data=average_results_df, palette='Blues', hue='Song', legend=False)
    plt.title('Average Pearson Correlation Across Participants for Each Song')
    plt.ylabel('Average Pearson Correlation Coefficient')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Visualization: Bar plot for average MCD score per song
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Song', y='Average MCD Score', data=average_results_df, palette='Greens', hue='Song', legend=False)
    plt.title('Average Mel-Cepstral Distortion Across Participants for Each Song')
    plt.ylabel('Average MCD Score')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare original and GAN-generated audio.")
    parser.add_argument('--original_folder', type=str, required=True, help='Directory containing the original audio files.')
    parser.add_argument('--gan_folder', type=str, required=True, help='Directory containing the GAN-generated audio files.')

    args = parser.parse_args()

    analyze_audio_folders(args.original_folder, args.gan_folder)

if __name__ == "__main__":
    main()


# python evaluate_audio.py --original_folder '/path/to/original_folder' --gan_folder '/path/to/gan_folder'
