"""
Semantic Encoding Aesthetics MCP Server

Translates text through various encoding systems (Morse, Braille, Dot Matrix)
with sentiment-driven visual parameters and optional color palette extraction.
"""

from fastmcp import FastMCP
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import base64
import io
from typing import Optional, Tuple, List, Dict
import re

# Sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Audio primitives
try:
    from .audio_primitives import (
        load_audio_file,
        analyze_audio_features,
        extract_waveform_segments,
        waveform_to_svg_path,
        generate_audio_sentiment_map,
        synthesize_audio_from_encoding,
        audio_to_base64
    )
    AUDIO_PRIMITIVES_AVAILABLE = True
except ImportError:
    AUDIO_PRIMITIVES_AVAILABLE = False

mcp = FastMCP("semantic-encoding-aesthetics")

# Encoding dictionaries
MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.--',
    '/': '-..-.', '(': '-.--.', ')': '-.--.-', '&': '.-...', ':': '---...',
    ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-', '_': '..--.-',
    '"': '.-..-.', '$': '...-..-', '@': '.--.-.', ' ': '/'
}

# Simplified Braille (Grade 1) - representing as dot patterns
BRAILLE_PATTERNS = {
    'A': '⠁', 'B': '⠃', 'C': '⠉', 'D': '⠙', 'E': '⠑', 'F': '⠋',
    'G': '⠛', 'H': '⠓', 'I': '⠊', 'J': '⠚', 'K': '⠅', 'L': '⠇',
    'M': '⠍', 'N': '⠝', 'O': '⠕', 'P': '⠏', 'Q': '⠟', 'R': '⠗',
    'S': '⠎', 'T': '⠞', 'U': '⠥', 'V': '⠧', 'W': '⠺', 'X': '⠭',
    'Y': '⠽', 'Z': '⠵', ' ': '⠀',
    '0': '⠚', '1': '⠁', '2': '⠃', '3': '⠉', '4': '⠙', '5': '⠑',
    '6': '⠋', '7': '⠛', '8': '⠓', '9': '⠊'
}


def encode_morse(text: str) -> str:
    """Convert text to Morse code."""
    return ' '.join(MORSE_CODE.get(c.upper(), '') for c in text)


def encode_braille(text: str) -> str:
    """Convert text to Braille patterns."""
    return ''.join(BRAILLE_PATTERNS.get(c.upper(), c) for c in text)


def encode_dot_matrix(text: str) -> Dict[str, any]:
    """
    Represent text as dot matrix grid information.
    Returns character count and grid dimensions.
    """
    chars = len(text)
    # Standard dot matrix: 5x7 character cell, plus spacing
    char_width = 6  # 5 dots + 1 space
    char_height = 9  # 7 dots + 2 vertical spacing
    
    # Estimate grid for wrapping at ~80 chars
    cols = min(80, chars)
    rows = (chars + cols - 1) // cols
    
    return {
        'character_count': chars,
        'grid_cols': cols,
        'grid_rows': rows,
        'pixel_width': cols * char_width,
        'pixel_height': rows * char_height,
        'char_cell_size': '5x7',
        'text': text
    }


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment using VADER.
    Returns polarity, arousal approximation, and compound score.
    """
    if not VADER_AVAILABLE:
        # Fallback to simple heuristics
        positive_words = ['good', 'great', 'excellent', 'happy', 'joy', 'love']
        negative_words = ['bad', 'terrible', 'sad', 'hate', 'angry', 'fear']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            polarity = 0.0
        else:
            polarity = (pos_count - neg_count) / total
        
        # Rough arousal based on punctuation and caps
        exclamations = text.count('!') + text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        arousal = min(1.0, (exclamations * 0.2 + caps_ratio) * 2)
        
        return {
            'polarity': polarity,
            'arousal': arousal,
            'valence': abs(polarity),
            'compound': polarity
        }
    
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    # VADER gives us compound, pos, neg, neu
    # Map to our dimensions
    polarity = scores['compound']  # -1 to 1
    
    # Arousal approximation: strength of emotion (how far from neutral)
    arousal = abs(scores['compound'])
    
    # Valence: emotional positivity
    valence = scores['pos'] - scores['neg']
    
    return {
        'polarity': polarity,
        'arousal': arousal,
        'valence': valence,
        'compound': scores['compound'],
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu']
    }


def extract_palette_from_image(image_data: str, n_colors: int = 5) -> List[Dict[str, any]]:
    """
    Extract dominant color palette from image.
    Accepts base64 encoded image data or file path.
    """
    try:
        # Try to decode as base64
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',', 1)[1]
        
        try:
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))
        except:
            # Try as file path
            img = Image.open(image_data)
        
        # Convert to RGB
        img = img.convert('RGB')
        
        # Resize for performance (max 200px on longest side)
        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
        
        # Get pixel data
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and their prevalence
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Calculate prevalence
        palette = []
        for i, color in enumerate(colors):
            prevalence = np.sum(labels == i) / len(labels)
            
            # Calculate luminance for sorting/categorization
            luminance = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) / 255
            
            palette.append({
                'rgb': tuple(color.tolist()),
                'hex': '#{:02x}{:02x}{:02x}'.format(*color),
                'prevalence': float(prevalence),
                'luminance': float(luminance)
            })
        
        # Sort by prevalence
        palette.sort(key=lambda x: x['prevalence'], reverse=True)
        
        return palette
        
    except Exception as e:
        return [{
            'error': f'Failed to extract palette: {str(e)}',
            'fallback': True
        }]


def map_sentiment_to_parameters(
    sentiment: Dict[str, float],
    dot_size_range: Tuple[float, float],
    spacing_range: Tuple[float, float],
    degradation_range: Tuple[float, float],
    sentiment_influence: float
) -> Dict[str, any]:
    """
    Map sentiment scores to visual parameters within user-defined ranges.
    """
    # Normalize sentiment influence (0-1)
    influence = max(0.0, min(1.0, sentiment_influence))
    
    # Map polarity (-1 to 1) to dot size range
    # Positive = larger, negative = smaller
    polarity_norm = (sentiment['polarity'] + 1) / 2  # 0 to 1
    dot_size_multiplier = (
        dot_size_range[0] + 
        (dot_size_range[1] - dot_size_range[0]) * polarity_norm * influence +
        (dot_size_range[0] + dot_size_range[1]) / 2 * (1 - influence)
    )
    
    # Map arousal (0 to 1) to spacing irregularity
    spacing_multiplier = (
        spacing_range[0] + 
        (spacing_range[1] - spacing_range[0]) * sentiment['arousal'] * influence +
        (spacing_range[0] + spacing_range[1]) / 2 * (1 - influence)
    )
    
    # Map valence to degradation (higher emotion = more degradation)
    degradation_amount = (
        degradation_range[0] + 
        (degradation_range[1] - degradation_range[0]) * sentiment['valence'] * influence
    )
    
    # Calculate rhythm/timing based on compound sentiment
    # Negative = slower, positive = faster
    base_timing = 1000  # ms
    timing_variation = sentiment['compound'] * 0.3 * influence
    timing_ms = base_timing * (1 - timing_variation)
    
    return {
        'dot_size_multiplier': round(dot_size_multiplier, 3),
        'spacing_multiplier': round(spacing_multiplier, 3),
        'degradation_amount': round(degradation_amount, 3),
        'timing_ms': round(timing_ms, 0),
        'sentiment_influence_applied': influence,
        'interpretation': {
            'polarity_effect': 'larger dots' if sentiment['polarity'] > 0 else 'smaller dots',
            'arousal_effect': 'irregular spacing' if sentiment['arousal'] > 0.5 else 'regular spacing',
            'emotional_intensity': 'high' if sentiment['valence'] > 0.6 else 'moderate' if sentiment['valence'] > 0.3 else 'subtle'
        }
    }


@mcp.tool()
def analyze_text_encoding(
    text: str,
    encoding_type: str = "dot_matrix",
    sentiment_influence: float = 0.7,
    dot_size_range: Tuple[float, float] = (1.0, 3.0),
    spacing_range: Tuple[float, float] = (0.8, 1.5),
    degradation_range: Tuple[float, float] = (0.0, 0.5),
    color_reference_image: Optional[str] = None,
    palette_size: int = 5,
    audio_primitive_source: Optional[str] = None,
    audio_segment_duration_ms: float = 50.0
) -> dict:
    """
    Analyze text with semantic encoding and sentiment-driven visual parameters.
    
    Args:
        text: Input text to encode and analyze
        encoding_type: Type of encoding - "morse", "braille", or "dot_matrix"
        sentiment_influence: How much sentiment affects visuals (0.0-1.0)
        dot_size_range: Min and max multipliers for dot/character size
        spacing_range: Min and max multipliers for spacing
        degradation_range: Min and max for visual degradation/weathering (0.0-1.0)
        color_reference_image: Optional base64 image data or file path for palette extraction
        palette_size: Number of dominant colors to extract (default 5)
        audio_primitive_source: Optional audio file path or base64 for waveform primitives
        audio_segment_duration_ms: Duration of each waveform segment in milliseconds (default 50)
    
    Returns:
        Complete analysis including encoding, sentiment, visual parameters, and optional palette/audio
    """
    # Validate encoding type
    encoding_type = encoding_type.lower()
    if encoding_type not in ['morse', 'braille', 'dot_matrix']:
        return {'error': f'Invalid encoding_type: {encoding_type}. Use "morse", "braille", or "dot_matrix"'}
    
    # Encode text
    if encoding_type == 'morse':
        encoded = encode_morse(text)
        encoding_info = {
            'type': 'morse',
            'encoded_text': encoded,
            'character_count': len(text),
            'symbol_count': len(encoded.replace(' ', ''))
        }
    elif encoding_type == 'braille':
        encoded = encode_braille(text)
        encoding_info = {
            'type': 'braille',
            'encoded_text': encoded,
            'character_count': len(text),
            'pattern_count': len(encoded)
        }
    else:  # dot_matrix
        encoding_info = encode_dot_matrix(text)
        encoding_info['type'] = 'dot_matrix'
    
    # Analyze sentiment
    sentiment = analyze_sentiment(text)
    
    # Map to visual parameters
    visual_params = map_sentiment_to_parameters(
        sentiment,
        dot_size_range,
        spacing_range,
        degradation_range,
        sentiment_influence
    )
    
    # Extract color palette if provided
    color_palette = None
    if color_reference_image:
        color_palette = extract_palette_from_image(color_reference_image, palette_size)
    
    # Process audio primitive if provided
    audio_data = None
    if audio_primitive_source and AUDIO_PRIMITIVES_AVAILABLE:
        try:
            # Load and analyze audio
            audio_waveform, sample_rate = load_audio_file(audio_primitive_source)
            audio_features = analyze_audio_features(audio_waveform, sample_rate)
            
            # Generate audio-text sentiment comparison
            audio_sentiment_map = generate_audio_sentiment_map(audio_features, sentiment)
            
            # Extract waveform segments for visualization
            num_segments = min(len(text), 100)  # One segment per character, up to 100
            segments = extract_waveform_segments(
                audio_waveform,
                sample_rate,
                num_segments,
                audio_segment_duration_ms
            )
            
            # Convert first few segments to SVG paths for preview
            segment_paths = []
            for i, segment in enumerate(segments[:20]):  # Preview first 20
                svg_path = waveform_to_svg_path(segment, width=100, height=50)
                segment_paths.append({
                    'index': i,
                    'svg_path': svg_path
                })
            
            audio_data = {
                'features': audio_features,
                'sentiment_comparison': audio_sentiment_map,
                'num_segments': len(segments),
                'segment_duration_ms': audio_segment_duration_ms,
                'preview_paths': segment_paths,
                'synesthetic_params': {
                    'frequency_multiplier': 1 + sentiment['polarity'] * 0.5,
                    'tempo_multiplier': 1 + sentiment['arousal'] * 0.5,
                    'amplitude': 0.5 + sentiment['valence'] * 0.5
                }
            }
            
        except Exception as e:
            audio_data = {'error': f'Failed to process audio: {str(e)}'}
    elif audio_primitive_source and not AUDIO_PRIMITIVES_AVAILABLE:
        audio_data = {'error': 'Audio processing requires librosa and soundfile. Install with: pip install librosa soundfile'}
    
    # Build complete analysis
    result = {
        'original_text': text,
        'encoding': encoding_info,
        'sentiment_analysis': sentiment,
        'visual_parameters': visual_params,
        'user_constraints': {
            'dot_size_range': dot_size_range,
            'spacing_range': spacing_range,
            'degradation_range': degradation_range,
            'sentiment_influence': sentiment_influence
        }
    }
    
    if color_palette:
        result['color_palette'] = color_palette
    
    if audio_data:
        result['audio_primitive'] = audio_data
    
    return result


@mcp.tool()
def generate_enhanced_prompt(
    analysis: dict,
    style_preset: str = "default",
    output_format: str = "svg"
) -> str:
    """
    Generate an enhanced prompt for Claude to synthesize the final artwork.
    
    Args:
        analysis: Output from analyze_text_encoding
        style_preset: Style direction - "default", "vintage", "clinical", "expressive"
        output_format: Desired output format - "svg", "canvas", "description"
    
    Returns:
        Enhanced prompt string for Claude to generate the artwork
    """
    text = analysis['original_text']
    encoding = analysis['encoding']
    sentiment = analysis['sentiment_analysis']
    visual = analysis['visual_parameters']
    constraints = analysis['user_constraints']
    
    # Build the prompt
    prompt_parts = []
    
    # Header
    prompt_parts.append(f"Generate a visual artwork using {encoding['type']} aesthetics with the following constraints:\n")
    
    # Text encoding section
    prompt_parts.append("TEXT ENCODING:")
    prompt_parts.append(f"- Original: \"{text}\"")
    if encoding['type'] == 'morse':
        prompt_parts.append(f"- Morse encoded: {encoding['encoded_text']}")
        prompt_parts.append(f"- {encoding['character_count']} characters, {encoding['symbol_count']} symbols")
    elif encoding['type'] == 'braille':
        prompt_parts.append(f"- Braille encoded: {encoding['encoded_text']}")
        prompt_parts.append(f"- {encoding['character_count']} characters, {encoding['pattern_count']} patterns")
    else:  # dot_matrix
        prompt_parts.append(f"- Dot matrix grid: {encoding['grid_cols']}x{encoding['grid_rows']} characters")
        prompt_parts.append(f"- Character cell: {encoding['char_cell_size']} dots")
        prompt_parts.append(f"- Estimated dimensions: {encoding['pixel_width']}x{encoding['pixel_height']} pixels")
    
    prompt_parts.append("")
    
    # Sentiment section
    prompt_parts.append("SENTIMENT ANALYSIS:")
    prompt_parts.append(f"- Polarity: {sentiment['polarity']:.2f} ({'positive' if sentiment['polarity'] > 0 else 'negative' if sentiment['polarity'] < 0 else 'neutral'})")
    prompt_parts.append(f"- Arousal: {sentiment['arousal']:.2f} ({'high energy' if sentiment['arousal'] > 0.6 else 'moderate' if sentiment['arousal'] > 0.3 else 'calm'})")
    prompt_parts.append(f"- Valence: {sentiment['valence']:.2f} ({'strong emotion' if sentiment['valence'] > 0.6 else 'moderate emotion' if sentiment['valence'] > 0.3 else 'subtle'})")
    prompt_parts.append(f"- Compound score: {sentiment['compound']:.2f}")
    
    prompt_parts.append("")
    
    # Visual parameters section
    prompt_parts.append("VISUAL PARAMETERS:")
    base_dot = 2.0  # Assume 2px base
    actual_size = base_dot * visual['dot_size_multiplier']
    prompt_parts.append(f"- Base dot size: {actual_size:.1f}px (scaled {visual['dot_size_multiplier']:.2f}x within user range [{constraints['dot_size_range'][0]}-{constraints['dot_size_range'][1]}])")
    prompt_parts.append(f"  • Sentiment polarity {sentiment['polarity']:+.2f} creates {visual['interpretation']['polarity_effect']}")
    prompt_parts.append(f"  • User range interpretation: {constraints['dot_size_range'][0]}x = baseline, {constraints['dot_size_range'][1]}x = maximum emphasis")
    
    prompt_parts.append(f"- Spacing multiplier: {visual['spacing_multiplier']:.2f} (within user range [{constraints['spacing_range'][0]}-{constraints['spacing_range'][1]}])")
    prompt_parts.append(f"  • Creates {visual['interpretation']['arousal_effect']}")
    
    prompt_parts.append(f"- Degradation: {visual['degradation_amount']*100:.0f}% (within user range [{constraints['degradation_range'][0]*100:.0f}%-{constraints['degradation_range'][1]*100:.0f}%])")
    prompt_parts.append(f"- Timing pulse: {visual['timing_ms']:.0f}ms intervals")
    prompt_parts.append(f"- Emotional intensity: {visual['interpretation']['emotional_intensity']}")
    
    prompt_parts.append("")
    
    # Color palette section
    if 'color_palette' in analysis and not analysis['color_palette'][0].get('error'):
        prompt_parts.append("COLOR PALETTE (extracted from reference image):")
        for i, color in enumerate(analysis['color_palette'], 1):
            rgb = color['rgb']
            prompt_parts.append(f"{i}. rgb{rgb} / {color['hex']} - {color['prevalence']*100:.0f}% prevalence, luminance {color['luminance']:.2f}")
        
        prompt_parts.append("")
        prompt_parts.append("SENTIMENT → COLOR MAPPING:")
        
        # Suggest color usage based on sentiment
        if sentiment['polarity'] > 0.3:
            prompt_parts.append("- Positive polarity suggests warm/light colors for primary elements")
        elif sentiment['polarity'] < -0.3:
            prompt_parts.append("- Negative polarity suggests cool/dark colors for primary elements")
        
        if sentiment['arousal'] > 0.6:
            prompt_parts.append("- High arousal: use contrasting colors for rhythm variations")
        
        prompt_parts.append("- Degradation effects should blend toward darker palette colors")
        prompt_parts.append("")
        prompt_parts.append("CONSTRAINT: All visual elements must use only these extracted colors.")
        prompt_parts.append("No gradients between non-palette colors. Dithering/halftoning allowed.")
    
    prompt_parts.append("")
    
    # Style directive
    style_directives = {
        'default': "Render with balanced emphasis on readability and emotional expression.",
        'vintage': "Emphasize nostalgic, weathered aesthetics with prominent degradation and mechanical imperfections.",
        'clinical': "Prioritize precision and clarity with minimal degradation and consistent spacing.",
        'expressive': "Maximize emotional impact through dramatic scale variations and bold degradation patterns."
    }
    
    prompt_parts.append("STYLE DIRECTIVE:")
    prompt_parts.append(style_directives.get(style_preset, style_directives['default']))
    
    # Add context based on sentiment
    if abs(sentiment['polarity']) > 0.5:
        tone = "resilience and strength" if sentiment['polarity'] > 0 else "tension and melancholy"
        prompt_parts.append(f"The sentiment suggests {tone} - consider how the encoding method reinforces this through visual weight and rhythm.")
    
    prompt_parts.append("")
    
    # Output format
    format_specs = {
        'svg': "SVG with embedded animation",
        'canvas': "HTML5 Canvas with JavaScript animation",
        'description': "Detailed written description of the artwork for implementation"
    }
    
    prompt_parts.append(f"OUTPUT FORMAT: {format_specs.get(output_format, format_specs['svg'])}, suitable for generative art display")
    
    return '\n'.join(prompt_parts)


@mcp.tool()
def synthesize_audio_from_text(
    text: str,
    encoding_type: str = "morse",
    sentiment: Optional[dict] = None,
    sample_rate: int = 22050,
    base_frequency: float = 440.0,
    duration_per_unit: float = 0.1,
    output_format: str = "wav"
) -> dict:
    """
    Synthesize audio directly from text encoding with sentiment modulation.
    
    Creates audible representation of text encoded as Morse, Braille, or Dot Matrix.
    Sentiment parameters modulate frequency, tempo, and amplitude.
    
    Args:
        text: Input text to encode and sonify
        encoding_type: "morse", "braille", or "dot_matrix"
        sentiment: Optional sentiment dict (if not provided, will analyze text)
        sample_rate: Audio sample rate in Hz (default 22050)
        base_frequency: Base tone frequency in Hz (default 440 = A4)
        duration_per_unit: Duration per encoded unit in seconds (default 0.1)
        output_format: "wav" or "ogg" (default "wav")
    
    Returns:
        Dictionary with audio data (base64), duration, and synthesis parameters
    """
    if not AUDIO_PRIMITIVES_AVAILABLE:
        return {
            'error': 'Audio synthesis requires librosa and soundfile',
            'install': 'pip install librosa soundfile'
        }
    
    # Analyze sentiment if not provided
    if sentiment is None:
        sentiment = analyze_sentiment(text)
    
    # Encode text
    encoding_type = encoding_type.lower()
    if encoding_type not in ['morse', 'braille', 'dot_matrix']:
        return {'error': f'Invalid encoding_type: {encoding_type}'}
    
    if encoding_type == 'morse':
        encoded = encode_morse(text)
        encoding_info = {
            'type': 'morse',
            'encoded_text': encoded,
            'character_count': len(text)
        }
    elif encoding_type == 'braille':
        encoded = encode_braille(text)
        encoding_info = {
            'type': 'braille',
            'encoded_text': encoded,
            'character_count': len(text)
        }
    else:  # dot_matrix
        encoding_info = encode_dot_matrix(text)
        encoding_info['type'] = 'dot_matrix'
    
    try:
        # Synthesize audio
        audio_data, sr = synthesize_audio_from_encoding(
            encoding_info,
            sentiment,
            sample_rate,
            base_frequency,
            duration_per_unit
        )
        
        # Convert to base64
        audio_base64 = audio_to_base64(audio_data, sr, output_format)
        
        # Calculate duration
        duration_seconds = len(audio_data) / sr
        
        # Synthesis parameters
        synth_params = {
            'base_frequency': base_frequency,
            'modulated_frequency': base_frequency * (1 + sentiment['polarity'] * 0.5),
            'tempo_multiplier': 1 + sentiment['arousal'] * 0.5,
            'amplitude': 0.5 + sentiment['valence'] * 0.5,
            'duration_per_unit': duration_per_unit,
            'sample_rate': sr
        }
        
        return {
            'audio_data': audio_base64,
            'duration_seconds': duration_seconds,
            'encoding': encoding_info,
            'sentiment_applied': sentiment,
            'synthesis_parameters': synth_params,
            'format': output_format,
            'interpretation': {
                'pitch': 'higher' if sentiment['polarity'] > 0.3 else 'lower' if sentiment['polarity'] < -0.3 else 'neutral',
                'tempo': 'faster' if sentiment['arousal'] > 0.6 else 'slower' if sentiment['arousal'] < 0.4 else 'moderate',
                'volume': 'louder' if sentiment['valence'] > 0.6 else 'softer' if sentiment['valence'] < 0.4 else 'moderate'
            }
        }
        
    except Exception as e:
        return {'error': f'Audio synthesis failed: {str(e)}'}


# For local testing
def main():
    mcp.run()


if __name__ == "__main__":
    main()
