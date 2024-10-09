import os
import argparse
import json
import numpy as np
import librosa
import soundfile as sf
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and concatenate segments
def load_and_concatenate_segments(folder, segment_prefix, padded_shape, original_shape, db_min, db_max, segment_height, segment_width):
    stride_height = segment_height // 2
    stride_width = segment_width // 2

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    segment_files = sorted([f for f in os.listdir(folder) if f.startswith(segment_prefix)],
                           key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not segment_files:
        raise ValueError(f"No segment files found with prefix {segment_prefix} in folder {folder}.")

    segments = []
    for segment_file in segment_files:
        segment_path = os.path.join(folder, segment_file)
        segment_image = Image.open(segment_path).convert('L')
        segment = np.array(segment_image)
        segment = np.interp(segment, (0, 255), (db_min, db_max))
        segments.append(segment)

    concatenated_spectrogram = np.zeros(padded_shape)
    count_matrix = np.zeros(padded_shape)

    idx = 0
    for i in range(0, padded_shape[1] - segment_width + 1, stride_width):
        for j in range(0, padded_shape[0] - segment_height + 1, stride_height):
            if idx < len(segments):
                concatenated_spectrogram[j:j + segment_height, i:i + segment_width] += segments[idx]
                count_matrix[j:j + segment_height, i:i + segment_width] += 1
                idx += 1

    count_matrix[count_matrix == 0] = 1
    concatenated_spectrogram /= count_matrix
    concatenated_spectrogram = concatenated_spectrogram[:original_shape[0], :original_shape[1]]

    return concatenated_spectrogram

# Function to reconstruct waveform using Griffin-Lim algorithm
def reconstruct_waveform_griffinlim(magnitude_spectrogram, sr=44100, n_iter=32):
    return librosa.griffinlim(magnitude_spectrogram, n_iter=n_iter, hop_length=512, win_length=1024)

# Function to save waveform as a WAV file
def save_waveform(y, output_wav_file, sr=44100):
    y = y.astype(np.float32)
    y = np.clip(y, -1.0, 1.0)  # Clip values to prevent out-of-range issues
    sf.write(output_wav_file, y, sr, format='WAV', subtype='PCM_16') # Explicitly set format and subtype

# Function to normalize audio
def normalize_audio(y):
    max_val = np.max(np.abs(y))
    if max_val == 0:
        return y
    return y / max_val

def reconstruct_and_save_audio(generated_folder, metadata_file, output_folder, segment_height, segment_width, segment_prefix='C'):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    for participant, data in metadata.items():
        for song, song_data in data['songs'].items():
            num_segments = song_data['num_segments']
            song_title = song_data['title']
            original_length = song_data['original_length']
            original_sr = song_data['original_sr']
            padded_shape = song_data['padding']
            db_min = song_data['db_min']  # Extract dB min from metadata
            db_max = song_data['db_max']  # Extract dB max from metadata

            print(f"Reconstructing {song_title} for participant {participant}...")

            # Load and concatenate the generated audio segments with dB range
            concatenated_spectrogram = load_and_concatenate_segments(
                generated_folder, f'{participant}_{song}_{segment_prefix}', padded_shape, original_length, db_min, db_max, segment_height, segment_width
            )

            # Convert concatenated spectrogram from dB to amplitude
            concatenated_spectrogram_magnitude = librosa.db_to_amplitude(concatenated_spectrogram)

            # Reconstruct the audio from the concatenated segmented spectrogram
            reconstructed_audio_concatenated = reconstruct_waveform_griffinlim(concatenated_spectrogram_magnitude)
            reconstructed_audio_concatenated = normalize_audio(reconstructed_audio_concatenated)

            # Save the reconstructed waveform as a WAV file
            output_wav_file = os.path.join(output_folder, f"{participant}_{song_title}.wav")
            try:
                save_waveform(reconstructed_audio_concatenated, output_wav_file, original_sr)
            except Exception as e:
                print(f"Error saving audio file {output_wav_file}: {e}")

            # Save the concatenated spectrogram as an image (for debugging/visualization)
            output_image_file = os.path.join(output_folder, f"{participant}_{song_title}_spectrogram.png")
            concatenated_spectrogram_image = Image.fromarray(np.uint8(concatenated_spectrogram))
            concatenated_spectrogram_image.save(output_image_file)

            # Plot and show the concatenated spectrogram
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(concatenated_spectrogram, sr=44100, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Concatenated Spectrogram: {participant} - {song_title}')
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Reconstruct and save audio from spectrogram segments.")
    parser.add_argument('--generated_folder', type=str, required=True, help='Directory containing the generated audio segments.')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to the metadata JSON file.')
    parser.add_argument('--output_folder', type=str, required=True, help='Directory to save the reconstructed audio files.')
    parser.add_argument('--segment_prefix', type=str, default='C', help='Prefix for the segment files (default: C).')
    parser.add_argument('--segment_height', type=int, default=50, help='Height of spectrogram segments.')
    parser.add_argument('--segment_width', type=int, default=50, help='Width of spectrogram segments.')

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    reconstruct_and_save_audio(
        args.generated_folder,
        args.metadata_file,
        args.output_folder,
        args.segment_height,  # Pass segment_height
        args.segment_width,  # Pass segment_width
        segment_prefix=args.segment_prefix
    )

if __name__ == "__main__":
    main()



#python reconstruct_audio.py --generated_folder '/path/to/generated_folder' --metadata_file '/path/to/metadata.json' --output_folder '/path/to/output_folder' --segment_prefix 'C'
