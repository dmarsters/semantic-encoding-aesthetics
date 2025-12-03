"""
Audio Primitives for Semantic Encoding Aesthetics

Handles audio waveform analysis, visualization, and synthesis.
Enables synesthetic text-to-audiovisual experiences.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import base64
import io

# Audio processing
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


def load_audio_file(audio_source: str, sample_rate: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return waveform data.
    
    Args:
        audio_source: Path to audio file or base64 encoded audio data
        sample_rate: Target sample rate (default 22050 Hz)
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if not AUDIO_AVAILABLE:
        raise ImportError("librosa and soundfile required for audio processing. Install with: pip install librosa soundfile")
    
    try:
        # Try to decode as base64
        if audio_source.startswith('data:audio'):
            # Remove data URL prefix
            audio_source = audio_source.split(',', 1)[1]
        
        try:
            audio_bytes = base64.b64decode(audio_source)
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            if sr != sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
        except:
            # Try as file path
            audio_data, sr = librosa.load(audio_source, sr=sample_rate, mono=True)
        
        return audio_data, sample_rate
        
    except Exception as e:
        raise ValueError(f"Failed to load audio: {str(e)}")


def analyze_audio_features(audio_data: np.ndarray, sample_rate: int) -> Dict[str, any]:
    """
    Extract audio features for visualization and synthesis.
    
    Args:
        audio_data: Audio waveform as numpy array
        sample_rate: Sample rate of audio
    
    Returns:
        Dictionary of audio features
    """
    if not AUDIO_AVAILABLE:
        raise ImportError("librosa required for audio analysis")
    
    # Basic statistics
    duration = len(audio_data) / sample_rate
    rms = librosa.feature.rms(y=audio_data)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
    
    # Zero crossing rate (indicates noisiness)
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    
    # Tempo and beat
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    
    # Chroma features (pitch class)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    
    # MFCC (timbre)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    
    return {
        'duration': duration,
        'sample_rate': sample_rate,
        'length_samples': len(audio_data),
        'rms_energy': {
            'mean': rms_mean,
            'std': rms_std,
            'min': float(np.min(rms)),
            'max': float(np.max(rms))
        },
        'spectral_centroid': {
            'mean': float(np.mean(spectral_centroids)),
            'std': float(np.std(spectral_centroids))
        },
        'spectral_rolloff': {
            'mean': float(np.mean(spectral_rolloff))
        },
        'zero_crossing_rate': {
            'mean': float(np.mean(zcr))
        },
        'tempo': float(tempo),
        'num_beats': len(beats),
        'chroma_mean': chroma.mean(axis=1).tolist(),
        'mfcc_mean': mfcc.mean(axis=1).tolist()
    }


def extract_waveform_segments(
    audio_data: np.ndarray,
    sample_rate: int,
    num_segments: int,
    segment_duration_ms: float = 50.0
) -> List[np.ndarray]:
    """
    Extract waveform segments for use as visual primitives.
    
    Args:
        audio_data: Audio waveform
        sample_rate: Sample rate
        num_segments: Number of segments to extract
        segment_duration_ms: Duration of each segment in milliseconds
    
    Returns:
        List of audio segments
    """
    segment_samples = int(sample_rate * segment_duration_ms / 1000.0)
    total_samples = len(audio_data)
    
    if num_segments == 1:
        # Return entire waveform
        return [audio_data]
    
    # Extract evenly spaced segments
    segments = []
    step = max(1, (total_samples - segment_samples) // (num_segments - 1))
    
    for i in range(num_segments):
        start_idx = min(i * step, total_samples - segment_samples)
        end_idx = start_idx + segment_samples
        segment = audio_data[start_idx:end_idx]
        segments.append(segment)
    
    return segments


def waveform_to_svg_path(
    waveform: np.ndarray,
    width: float = 100.0,
    height: float = 50.0,
    normalize: bool = True
) -> str:
    """
    Convert waveform to SVG path string for visualization.
    
    Args:
        waveform: Audio waveform segment
        width: SVG width
        height: SVG height
        normalize: Whether to normalize amplitude
    
    Returns:
        SVG path data string
    """
    if normalize and len(waveform) > 0:
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    
    # Sample points for visualization
    num_points = min(len(waveform), int(width))
    if len(waveform) > num_points:
        indices = np.linspace(0, len(waveform) - 1, num_points, dtype=int)
        waveform = waveform[indices]
    
    # Convert to SVG coordinates
    x_coords = np.linspace(0, width, len(waveform))
    y_coords = height / 2 - (waveform * height / 2)
    
    # Build path string
    path = f"M {x_coords[0]:.2f} {y_coords[0]:.2f}"
    for x, y in zip(x_coords[1:], y_coords[1:]):
        path += f" L {x:.2f} {y:.2f}"
    
    return path


def generate_audio_sentiment_map(
    audio_features: Dict[str, any],
    text_sentiment: Dict[str, float]
) -> Dict[str, any]:
    """
    Map audio features to sentiment-like dimensions for comparison with text sentiment.
    
    Audio "sentiment" interpretation:
    - High RMS energy → High arousal
    - High spectral centroid (brightness) → Positive polarity
    - High tempo → High arousal
    - Complex timbre (MFCC variance) → High valence
    
    Args:
        audio_features: Audio analysis results
        text_sentiment: Text sentiment analysis results
    
    Returns:
        Audio sentiment mapping and comparison
    """
    # Normalize features to 0-1 ranges (rough heuristics)
    
    # RMS energy typically 0-0.5 for normalized audio
    energy_normalized = min(1.0, audio_features['rms_energy']['mean'] / 0.3)
    
    # Spectral centroid typically 500-4000 Hz
    brightness_normalized = min(1.0, 
        (audio_features['spectral_centroid']['mean'] - 500) / 3500)
    
    # Tempo typically 60-180 BPM
    tempo_normalized = min(1.0, (audio_features['tempo'] - 60) / 120)
    
    # Zero crossing rate typically 0-0.5
    noisiness_normalized = min(1.0, audio_features['zero_crossing_rate']['mean'] / 0.3)
    
    # Derive audio "sentiment"
    audio_polarity = (brightness_normalized * 2 - 1)  # -1 to 1
    audio_arousal = (energy_normalized + tempo_normalized) / 2  # 0 to 1
    audio_valence = energy_normalized  # 0 to 1
    
    # Compare with text sentiment
    polarity_alignment = 1 - abs(audio_polarity - text_sentiment['polarity']) / 2
    arousal_alignment = 1 - abs(audio_arousal - text_sentiment['arousal'])
    valence_alignment = 1 - abs(audio_valence - text_sentiment['valence'])
    
    overall_alignment = (polarity_alignment + arousal_alignment + valence_alignment) / 3
    
    return {
        'audio_sentiment': {
            'polarity': audio_polarity,
            'arousal': audio_arousal,
            'valence': audio_valence,
            'interpretation': {
                'energy': 'high' if energy_normalized > 0.6 else 'moderate' if energy_normalized > 0.3 else 'low',
                'brightness': 'bright' if brightness_normalized > 0.6 else 'moderate' if brightness_normalized > 0.3 else 'dark',
                'tempo': 'fast' if tempo_normalized > 0.6 else 'moderate' if tempo_normalized > 0.3 else 'slow',
                'character': 'energetic' if audio_arousal > 0.6 else 'calm' if audio_arousal < 0.4 else 'neutral'
            }
        },
        'text_sentiment': text_sentiment,
        'alignment': {
            'polarity': polarity_alignment,
            'arousal': arousal_alignment,
            'valence': valence_alignment,
            'overall': overall_alignment,
            'interpretation': 'high alignment' if overall_alignment > 0.7 else 'moderate alignment' if overall_alignment > 0.4 else 'divergent'
        },
        'synthesis_recommendation': {
            'use_audio_tempo': audio_features['tempo'],
            'use_audio_rhythm': audio_features['num_beats'] > 0,
            'blend_weight': overall_alignment,  # Higher alignment = trust audio more
            'highlight_contrast': overall_alignment < 0.4  # Low alignment = show divergence
        }
    }


def synthesize_audio_from_encoding(
    encoding_data: Dict[str, any],
    sentiment: Dict[str, float],
    sample_rate: int = 22050,
    base_frequency: float = 440.0,
    duration_per_unit: float = 0.1
) -> Tuple[np.ndarray, int]:
    """
    Synthesize audio from text encoding and sentiment.
    
    Args:
        encoding_data: Text encoding information
        sentiment: Sentiment analysis results
        sample_rate: Audio sample rate
        base_frequency: Base frequency in Hz (A4 = 440)
        duration_per_unit: Duration per encoded unit in seconds
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if encoding_data['type'] == 'morse':
        # Morse code audio synthesis
        return synthesize_morse_audio(
            encoding_data['encoded_text'],
            sentiment,
            sample_rate,
            base_frequency,
            duration_per_unit
        )
    
    elif encoding_data['type'] == 'braille':
        # Braille as tone patterns (6 dots = 6 tone positions)
        return synthesize_braille_audio(
            encoding_data['encoded_text'],
            sentiment,
            sample_rate,
            base_frequency
        )
    
    else:  # dot_matrix
        # Dot matrix as rhythmic patterns
        return synthesize_dotmatrix_audio(
            encoding_data['text'],
            sentiment,
            sample_rate,
            base_frequency,
            duration_per_unit
        )


def synthesize_morse_audio(
    morse_code: str,
    sentiment: Dict[str, float],
    sample_rate: int,
    base_frequency: float,
    unit_duration: float
) -> Tuple[np.ndarray, int]:
    """
    Synthesize Morse code as audio tones.
    
    Dit = 1 unit, Dah = 3 units
    Space between elements = 1 unit
    Space between letters = 3 units
    Space between words = 7 units
    """
    # Sentiment modulation
    frequency = base_frequency * (1 + sentiment['polarity'] * 0.5)  # ±50% frequency shift
    tempo_multiplier = 1 + sentiment['arousal'] * 0.5  # Faster when more aroused
    
    dit_duration = unit_duration / tempo_multiplier
    samples_per_unit = int(sample_rate * dit_duration)
    
    audio_segments = []
    
    for char in morse_code:
        if char == '.':  # Dit
            t = np.linspace(0, dit_duration, samples_per_unit)
            tone = np.sin(2 * np.pi * frequency * t)
            # Apply envelope to avoid clicks
            envelope = np.ones_like(tone)
            fade_samples = int(samples_per_unit * 0.1)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            audio_segments.append(tone * envelope)
            audio_segments.append(np.zeros(samples_per_unit))  # Gap after dit
            
        elif char == '-':  # Dah
            t = np.linspace(0, dit_duration * 3, samples_per_unit * 3)
            tone = np.sin(2 * np.pi * frequency * t)
            envelope = np.ones_like(tone)
            fade_samples = int(len(tone) * 0.1)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            audio_segments.append(tone * envelope)
            audio_segments.append(np.zeros(samples_per_unit))  # Gap after dah
            
        elif char == ' ':  # Letter space (already have 1 unit, add 2 more)
            audio_segments.append(np.zeros(samples_per_unit * 2))
            
        elif char == '/':  # Word space (already have 1 unit, add 6 more)
            audio_segments.append(np.zeros(samples_per_unit * 6))
    
    # Concatenate and normalize
    audio_data = np.concatenate(audio_segments) if audio_segments else np.array([])
    
    # Apply sentiment-driven amplitude modulation
    amplitude = 0.5 + sentiment['valence'] * 0.5  # 0.5 to 1.0
    audio_data = audio_data * amplitude
    
    return audio_data.astype(np.float32), sample_rate


def synthesize_braille_audio(
    braille_text: str,
    sentiment: Dict[str, float],
    sample_rate: int,
    base_frequency: float
) -> Tuple[np.ndarray, int]:
    """
    Synthesize Braille as multi-tone chords.
    Each Braille cell has 6 dots - represent as 6-note chord.
    """
    # Sentiment modulation
    tempo_multiplier = 1 + sentiment['arousal'] * 0.5
    note_duration = 0.2 / tempo_multiplier
    samples_per_note = int(sample_rate * note_duration)
    
    # Define frequencies for 6 dot positions (pentatonic scale)
    dot_frequencies = [
        base_frequency * (1 + sentiment['polarity'] * 0.3),  # Root
        base_frequency * 1.125,  # Major second
        base_frequency * 1.25,   # Major third
        base_frequency * 1.5,    # Perfect fifth
        base_frequency * 1.667,  # Major sixth
        base_frequency * 2.0     # Octave
    ]
    
    audio_segments = []
    
    for char in braille_text:
        # Generate a short chord for each character
        t = np.linspace(0, note_duration, samples_per_note)
        chord = np.zeros(samples_per_note)
        
        # Simple approach: use character code to determine which frequencies to use
        char_code = ord(char)
        for i, freq in enumerate(dot_frequencies):
            if (char_code >> i) & 1:  # Bit is set
                chord += np.sin(2 * np.pi * freq * t)
        
        # Normalize and apply envelope
        if np.max(np.abs(chord)) > 0:
            chord = chord / np.max(np.abs(chord))
        
        envelope = np.ones_like(chord)
        fade_samples = int(samples_per_note * 0.1)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        audio_segments.append(chord * envelope * 0.5)
        audio_segments.append(np.zeros(samples_per_note // 4))  # Small gap
    
    audio_data = np.concatenate(audio_segments) if audio_segments else np.array([])
    
    # Apply sentiment amplitude
    amplitude = 0.5 + sentiment['valence'] * 0.5
    audio_data = audio_data * amplitude
    
    return audio_data.astype(np.float32), sample_rate


def synthesize_dotmatrix_audio(
    text: str,
    sentiment: Dict[str, float],
    sample_rate: int,
    base_frequency: float,
    unit_duration: float
) -> Tuple[np.ndarray, int]:
    """
    Synthesize dot matrix as percussive/rhythmic audio.
    Each character becomes a short click/blip.
    """
    # Sentiment modulation
    tempo_multiplier = 1 + sentiment['arousal'] * 0.5
    click_duration = unit_duration * 0.1 / tempo_multiplier
    gap_duration = unit_duration * 0.9 / tempo_multiplier
    
    samples_per_click = int(sample_rate * click_duration)
    samples_per_gap = int(sample_rate * gap_duration)
    
    frequency = base_frequency * 2 * (1 + sentiment['polarity'] * 0.5)
    
    audio_segments = []
    
    for char in text:
        # Generate short blip for each character
        t = np.linspace(0, click_duration, samples_per_click)
        
        if char == ' ':
            # Longer silence for spaces
            audio_segments.append(np.zeros(samples_per_gap * 2))
        else:
            # Sharp attack, exponential decay
            blip = np.sin(2 * np.pi * frequency * t)
            decay = np.exp(-t * 20)  # Exponential decay
            blip = blip * decay
            
            audio_segments.append(blip)
            audio_segments.append(np.zeros(samples_per_gap))
    
    audio_data = np.concatenate(audio_segments) if audio_segments else np.array([])
    
    # Apply sentiment amplitude
    amplitude = 0.5 + sentiment['valence'] * 0.5
    audio_data = audio_data * amplitude
    
    return audio_data.astype(np.float32), sample_rate


def audio_to_base64(audio_data: np.ndarray, sample_rate: int, format: str = 'wav') -> str:
    """
    Convert audio data to base64 string for embedding or transmission.
    
    Args:
        audio_data: Audio waveform
        sample_rate: Sample rate
        format: Audio format ('wav' or 'ogg')
    
    Returns:
        Base64 encoded audio string with data URL prefix
    """
    if not AUDIO_AVAILABLE:
        raise ImportError("soundfile required for audio encoding")
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format=format.upper())
    buffer.seek(0)
    
    audio_bytes = buffer.read()
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    
    mime_type = f'audio/{format}'
    return f'data:{mime_type};base64,{base64_audio}'
