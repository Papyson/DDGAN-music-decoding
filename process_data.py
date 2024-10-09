import argparse
import os
import json
import numpy as np
import librosa
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import mne
from mne_bids import BIDSPath, read_raw_bids

# Define paths and event mapping
audio_files = {
    'Song_One': 'adele_someone_like_you_clip.wav',
    'Song_Two': 'beethoven_symphony_no_5_clip.wav',
    'Song_Three': 'john_coltrane_giant_steps_clip.wav',
    'Song_Four': 'led_zeppelin_stairway_to_heaven_clip.wav',
    'Song_Five': 'the_beatles_yesterday_clip.wav'
}

def get_principal_component(evoked):
    evoked_data = evoked.data.T  # Transpose to shape (n_times, n_channels)
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(evoked_data).flatten()
    return principal_component

def eeg_to_spectrogram(eeg_data, sr=500):
    eeg_data = (eeg_data - np.mean(eeg_data)) / np.std(eeg_data)
    S = librosa.stft(eeg_data, n_fft=1024, hop_length=6)
    S_magnitude = np.abs(S)
    S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max)
    S_db_normalized = np.interp(S_db, (S_db.min(), S_db.max()), (0, 255))
    return S_db_normalized.astype(np.uint8)

def audio_to_spectrogram(audio_signal, sr=44100):
    audio_signal = (audio_signal - np.mean(audio_signal)) / np.std(audio_signal)
    S = librosa.stft(audio_signal, n_fft=1024, hop_length=512)
    S_magnitude = np.abs(S)
    S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max)
    db_min = S_db.min()
    db_max = S_db.max()
    S_db_normalized = np.interp(S_db, (S_db.min(), S_db.max()), (0, 255))
    return S_db_normalized.astype(np.uint8), db_min, db_max

def segment_spectrogram(spectrogram, segment_height, segment_width):
    stride_height = segment_height // 2
    stride_width = segment_width // 2
    pad_height = (segment_height - spectrogram.shape[0] % stride_height) % stride_height
    pad_width = (segment_width - spectrogram.shape[1] % stride_width) % stride_width
    padded_spectrogram = np.pad(spectrogram, ((0, pad_height), (0, pad_width)), mode='constant')
    segments = []
    for i in range(0, padded_spectrogram.shape[1] - segment_width + 1, stride_width):
        for j in range(0, padded_spectrogram.shape[0] - segment_height + 1, stride_height):
            segment = padded_spectrogram[j:j + segment_height, i:i + segment_width]
            segments.append(segment)
    return segments, padded_spectrogram.shape

def cascade_segments(eeg_segments, audio_segments, ratio=(3, 2), cascade_type='vertical'):
    transition_signals = []
    num_segments = min(len(eeg_segments), len(audio_segments))
    for i in range(num_segments):
        eeg_segment = eeg_segments[i]
        audio_segment = audio_segments[i]
        if cascade_type == 'horizontal':
            eeg_width = int(eeg_segment.shape[1] * (ratio[0] / sum(ratio)))
            audio_width = eeg_segment.shape[1] - eeg_width
            eeg_part = eeg_segment[:, :eeg_width]
            audio_part = audio_segment[:, :audio_width]
            combined_segment = np.concatenate((eeg_part, audio_part), axis=1)
        elif cascade_type == 'vertical':
            eeg_height = int(eeg_segment.shape[0] * (ratio[0] / sum(ratio)))
            audio_height = eeg_segment.shape[0] - eeg_height
            eeg_part = eeg_segment[:eeg_height, :]
            audio_part = audio_segment[:audio_height, :]
            combined_segment = np.concatenate((eeg_part, audio_part), axis=0)
        transition_signals.append(combined_segment)
    return transition_signals

def save_segments(segments, folder, prefix, participant, song):
    os.makedirs(folder, exist_ok=True)
    for i, segment in enumerate(segments):
        if isinstance(segment, np.ndarray):
            segment_image = Image.fromarray(segment)
        else:
            segment_image = segment
        segment_resized = segment_image.resize((50, 50), Image.ANTIALIAS)
        segment_filename = f'{participant}_{song}_{prefix}_{i+1}.png'
        segment_resized.save(os.path.join(folder, segment_filename))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def correct_annotations(raw):
    annotations = raw.annotations
    old_description = 'condition 5'
    new_description = 'ST SongTwo - Group 2'
    if not any(desc == new_description for desc in annotations.description):
        updated_onsets = []
        updated_durations = []
        updated_descriptions = []
        for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
            if description == old_description:
                updated_onsets.append(onset)
                updated_durations.append(duration)
                updated_descriptions.append(new_description)
            else:
                updated_onsets.append(onset)
                updated_durations.append(duration)
                updated_descriptions.append(description)
        updated_annotations = mne.Annotations(
            onset=updated_onsets,
            duration=updated_durations,
            description=updated_descriptions
        )
        raw.set_annotations(updated_annotations)

def process_participant(subject_id, bids_root, stimulus_folder, segment_height, segment_width):
    bids_path = BIDSPath(subject=subject_id, task='musiclisten', datatype='eeg', root=bids_root)
    raw = read_raw_bids(bids_path)
    picks = ['T7', 'T8', 'C4', 'C3']
    raw.pick_channels(picks)
    correct_annotations(raw)
    events, event_id = mne.events_from_annotations(raw)
    tmin, tmax = -2, 30
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(tmin, 0), preload=True)
    epochs.apply_baseline(baseline=(tmin, 0))
    evoked_responses = {
        'Song_One': epochs['ST SongOne - Group 1'].average(),
        'Song_Two': epochs['ST SongTwo - Group 2'].average(),
        'Song_Three': epochs['ST SongThree - Group 3'].average(),
        'Song_Four': epochs['ST SongFour - Group 4'].average(),
        'Song_Five': epochs['ST SongFive - Group 5'].average()
    }
    eeg_segments_list = []
    audio_segments_list = []
    transition_segments_list = []
    participant_metadata = {'participant_id': subject_id, 'songs': {}}
    for song, evoked in evoked_responses.items():
        principal_component = get_principal_component(evoked)
        eeg_spectrogram = eeg_to_spectrogram(principal_component, sr=500)
        eeg_segments, eeg_padding = segment_spectrogram(eeg_spectrogram, segment_height, segment_width)
        eeg_segments_list.extend(eeg_segments)
        audio_path = os.path.join(stimulus_folder, audio_files[song])
        audio_file, aud_sr = librosa.load(audio_path, sr=44100)
        audio_spectrogram, db_min, db_max = audio_to_spectrogram(audio_file, sr=aud_sr)
        audio_segments, audio_padding = segment_spectrogram(audio_spectrogram, segment_height, segment_width)
        audio_segments_list.extend(audio_segments)
        transition_signals = cascade_segments(eeg_segments, audio_segments)
        transition_segments_list.extend(transition_signals)
        participant_metadata['songs'][song] = {
            'title': song,
            'num_segments': len(audio_segments),
            'original_length': audio_spectrogram.shape,
            'original_sr': aud_sr,
            'padding': audio_padding,
            'db_min': db_min,
            'db_max': db_max
        }
    return eeg_segments_list, audio_segments_list, transition_segments_list, participant_metadata

def save_data_for_participants(participant_list, save_folder, bids_root, stimulus_folder, segment_height, segment_width):
    eeg_segments_list = []
    audio_segments_list = []
    transition_segments_list = []
    metadata = {}
    for participant in participant_list:
        eeg_segments, audio_segments, transition_segments, participant_metadata = process_participant(participant, bids_root, stimulus_folder, segment_height, segment_width)
        eeg_segments_list.extend(eeg_segments)
        audio_segments_list.extend(audio_segments)
        transition_segments_list.extend(transition_segments)
        metadata[participant] = participant_metadata
        for song, song_data in participant_metadata['songs'].items():
            song_prefix = song_data['title']
            num_segments = song_data['num_segments']
            save_segments(eeg_segments_list[:num_segments], os.path.join(save_folder, 'A'), 'A', participant, song_prefix)
            save_segments(audio_segments_list[:num_segments], os.path.join(save_folder, 'C'), 'C', participant, song_prefix)
            save_segments(transition_segments_list[:num_segments], os.path.join(save_folder, 'B'), 'B', participant, song_prefix)
            eeg_segments_list = eeg_segments_list[num_segments:]
            audio_segments_list = audio_segments_list[num_segments:]
            transition_segments_list = transition_segments_list[num_segments:]
    with open(os.path.join(save_folder, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, cls=NumpyEncoder, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process EEG and audio data for participants.")
    parser.add_argument('--bids_root', type=str, required=True, help='Root directory for BIDS data.')
    parser.add_argument('--stimulus_folder', type=str, required=True, help='Directory for audio stimulus files.')
    parser.add_argument('--segment_height', type=int, default=50, help='Height of spectrogram segments.')
    parser.add_argument('--segment_width', type=int, default=50, help='Width of spectrogram segments.')
    parser.add_argument('--train_save_folder', type=str, required=True, help='Directory to save training data.')
    parser.add_argument('--val_save_folder', type=str, required=True, help='Directory to save validation data.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of participants for validation set.')

    args = parser.parse_args()

    participants = [f's{i+1}' for i in range(23)]  # Generate participant IDs
    train_participants, val_participants = train_test_split(participants, test_size=args.test_size, random_state=42)

    save_data_for_participants(train_participants, args.train_save_folder, args.bids_root, args.stimulus_folder, args.segment_height, args.segment_width)
    save_data_for_participants(val_participants, args.val_save_folder, args.bids_root, args.stimulus_folder, args.segment_height, args.segment_width)

if __name__ == "__main__":
    main()


#python process_data.py --bids_root '/path/to/bids' --stimulus_folder '/path/to/stimuli' --train_save_folder '/path/to/train' --val_save_folder '/path/to/val'
