# Semantic Encoding Aesthetics

An MCP server that transforms text into visually and sonically expressive parameters by encoding it as Morse code, Braille, or Dot Matrix patterns, with sentiment-driven modulation of visual properties.

## Overview

Semantic Encoding Aesthetics bridges linguistic meaning and visual form by:

1. **Encoding text** as discrete symbolic systems (Morse, Braille, Dot Matrix)
2. **Analyzing sentiment** to extract emotional intensity, arousal, and valence
3. **Mapping sentiment to visual parameters** (dot size, spacing, degradation)
4. **Extracting color palettes** from reference images for coherent visual grounding
5. **Synthesizing audio** from encoded patterns with sentiment modulation

The result is a deterministic framework where text semantics directly influence visual and sonic aesthetics—not through prompt engineering, but through mathematical mapping of linguistic meaning to perceptual parameters.

## Architecture

### Three-Layer Pattern

**Layer 1: Text Analysis (Claude)**
- Parse user intent and text selection
- Analyze sentiment and emotional character
- Specify reference image or palette preferences

**Layer 2: Deterministic Mapping (Zero LLM Cost)**
- Encode text into discrete grid systems
- Extract dominant colors from reference images
- Map sentiment dimensions to visual parameters
- Generate false-color channel assignments (for audio)

**Layer 3: Creative Synthesis (Claude)**
- Synthesize parameters into narrative image prompts
- Generate enhanced prompts ready for image generation models
- Compose multi-modal outputs (visual + audio)

## Core Tools

### `analyze_text_encoding`
Encode text and extract visual/audio parameters with optional color palette extraction.

```python
analyze_text_encoding(
    text: str,
    encoding_type: str = "dot_matrix",  # morse, braille, dot_matrix
    sentiment_influence: float = 0.7,    # 0.0-1.0 (how much sentiment affects params)
    dot_size_range: list = [1.0, 3.0],
    spacing_range: list = [0.8, 1.5],
    degradation_range: list = [0.0, 0.5],
    color_reference_image: str = None,   # path or base64
    palette_size: int = 5,
    audio_primitive_source: str = None,  # path or base64
    audio_segment_duration_ms: int = 50
) → Dict
```

**Returns:**
- `encoding`: Character count, grid dimensions, pixel specifications
- `sentiment_analysis`: Polarity, arousal, valence, compound scores
- `visual_parameters`: Dot size multiplier, spacing multiplier, degradation amount
- `color_palette`: Extracted dominant colors with hex values
- `audio_data`: Optional waveform specifications

### `generate_enhanced_prompt`
Synthesize analysis into a complete image generation prompt.

```python
generate_enhanced_prompt(
    analysis: Dict,
    style_preset: str = "default",  # vintage, clinical, expressive, default
    output_format: str = "svg"      # svg, canvas, description
) → str
```

**Returns:** Enhanced prompt ready for Flux, Midjourney, DALL-E, or Stable Diffusion

### `synthesize_audio_from_text`
Create audible representation of encoded text with sentiment modulation.

```python
synthesize_audio_from_text(
    text: str,
    encoding_type: str = "morse",
    sentiment: Dict = None,
    sample_rate: int = 22050,
    base_frequency: float = 440,      # Hz (A4 = 440)
    duration_per_unit: float = 0.1,   # seconds per encoded unit
    output_format: str = "wav"        # wav, ogg
) → Dict
```

**Returns:** Base64-encoded audio data, duration, synthesis parameters

## Encoding Types

### Dot Matrix (5×7 Character Cell)
Traditional dot-matrix printer aesthetic. Each character rendered as a 5-wide, 7-tall grid of discrete dots. Creates pixelated, retro-computational visual quality.

- **Visual character:** Geometric, precise, grid-based
- **Emotional register:** Clinical, archival, nostalgic-technical
- **Typical use:** Vintage typography, data visualization, memorial/archival aesthetics

### Morse Code (· and −)
Rhythmic encoding alternating between dots (short signals) and dashes (long signals) with spaces between characters.

- **Visual character:** Linear, rhythmic, telegraph-era
- **Emotional register:** Historical, urgent, mysterious
- **Typical use:** Espionage, communication breakdown, temporal displacement

### Braille (2×3 Dot Patterns)
Tactile encoding system representing 64 characters through six-dot combinations. Can be rendered visually as raised dot patterns.

- **Visual character:** Organic, subtle, accessibility-forward
- **Emotional register:** Intimate, hidden, sensory-focused
- **Typical use:** Accessibility aesthetics, hidden messages, tactile/sensory compositions

## Sentiment-Driven Parameters

The server maps linguistic sentiment onto visual properties:

| Sentiment Dimension | Visual Effect | Range |
|---|---|---|
| **Polarity** (positive/negative) | Dot size | Smaller (negative) → Larger (positive) |
| **Arousal** (calm/excited) | Spacing regularity | Uniform (calm) → Irregular (excited) |
| **Valence** (unpleasant/pleasant) | Degradation/blur | High degradation (unpleasant) → Clean (pleasant) |
| **Emotional intensity** | Timing/rhythm | Slower (subdued) → Faster (intense) |

**Sentiment Influence Scale (0.0-1.0):**
- `0.0`: Neutral parameters regardless of text sentiment
- `0.5`: Moderate sentiment effect (balanced with encoding structure)
- `1.0`: Maximum sentiment modulation (emotional character dominates)

## Color Palette Extraction

When `color_reference_image` is provided, the server:

1. Analyzes dominant color regions
2. Extracts 3-7 primary colors (customizable via `palette_size`)
3. Maps colors to warm/cool/neutral categories
4. Returns hex values and semantic descriptors

**Use cases:**
- Ground dot-matrix text in specific color harmonies
- Extract palette from reference artwork, then apply to text encoding
- Create compositional coherence between text and visual context

## Audio Synthesis

Text can be sonified by:

1. Encoding as discrete symbols (Morse dots/dashes, Braille dot counts, Dot Matrix patterns)
2. Mapping symbols to frequency/amplitude values
3. Modulating tone qualities based on sentiment (frequency↑ for arousal, amplitude↓ for negative polarity)
4. Rendering as WAV or OGG

**Perceptual mapping:**
- **Morse:** Traditional telegraph tones (800-1200 Hz base)
- **Braille:** Dot count maps to harmonic series (fundamental + overtones)
- **Dot Matrix:** Grid density creates rhythmic patterns (faster grids = higher pitched repetitions)

## Usage Examples

### Example 1: Text to Image with Sentiment-Driven Degradation

```python
# Encode melancholic text
analysis = analyze_text_encoding(
    text="Our beloved life vests are melting!",
    encoding_type="dot_matrix",
    sentiment_influence=0.8,
    color_reference_image="reference_image.png",
    degradation_range=[0.1, 0.6]  # Allow significant decay
)

# Generate image prompt
prompt = generate_enhanced_prompt(
    analysis=analysis,
    style_preset="vintage",
    output_format="description"
)
```

### Example 2: Morse Code Sonification

```python
# Create audio from historical text
audio = synthesize_audio_from_text(
    text="SOS",
    encoding_type="morse",
    base_frequency=750,  # Lower, more dramatic tone
    duration_per_unit=0.2,  # Slower transmission
    output_format="wav"
)
# audio["data"] contains base64-encoded WAV
```

### Example 3: Palette-Extracted Dot Matrix

```python
# Extract palette from artwork, then encode text in those colors
analysis = analyze_text_encoding(
    text="And they never heard surf music again",
    encoding_type="dot_matrix",
    color_reference_image="surreal_composition.png",
    palette_size=5,
    sentiment_influence=0.0  # Neutral sentiment for finality
)
```

## Composition Patterns

### Sentiment × Reference Image

Combine text sentiment with visual reference to create coherent emotional ecosystems:

```
Positive + Vibrant Reference → Clean dots, irregular bright spacing
Negative + Muted Reference → Degraded dots, sparse tonal spacing
Neutral + Layered Reference → Precise grid, regular spacing
```

### Encoding Type × Emotional Tone

Different encodings express different emotional registers:

```
Morse + Urgent sentiment → Telegraph-speed urgency
Braille + Intimate sentiment → Tactile, hidden quality
Dot Matrix + Nostalgic sentiment → Retro-computational melancholy
```

### Multi-Modal Composition

Combine visual output with audio:

```
Text → Visual parameters (image generation)
Text → Audio parameters (sonification)
Result: Coordinated visual + sonic expression of same semantic content
```

## Parameter Ranges and Defaults

| Parameter | Range | Default | Notes |
|---|---|---|---|
| `sentiment_influence` | 0.0–1.0 | 0.7 | How much sentiment drives visual parameters |
| `dot_size_multiplier` | 1.0–3.0 | 2.3 | Rendered dot size relative to cell |
| `spacing_multiplier` | 0.8–1.5 | 1.145 | Character spacing regularity |
| `degradation_amount` | 0.0–0.5 | 0.0 | Visual decay/blur applied (0=pristine, 0.5=heavily degraded) |
| `base_frequency` | 100–4000 Hz | 440 | Audio base tone |
| `duration_per_unit` | 0.05–1.0 s | 0.1 | Audio duration per encoded symbol |
| `palette_size` | 3–7 | 5 | Number of dominant colors to extract |

## Workflow Integration

### For Image Generation
1. Analyze text with `analyze_text_encoding`
2. Generate prompt with `generate_enhanced_prompt`
3. Pass prompt to Flux, Midjourney, Stable Diffusion, etc.

### For ComfyUI Workflows
The enhanced prompt integrates directly into text nodes. Semantic parameters inform composition guidance without requiring separate nodes.

### For Data Visualization
Use encoding parameters to drive:
- Grid-based layouts (dot matrix character cells)
- Color assignments (extracted palettes)
- Temporal rhythm (sentiment-driven timing)

## Performance Characteristics

| Operation | Cost | Speed |
|---|---|---|
| Text encoding | $0 | <100ms |
| Sentiment analysis | $0 | <50ms |
| Palette extraction | $0 | <200ms (depends on image size) |
| Prompt synthesis | 1× Claude call | <2s |
| Audio synthesis | $0 | <500ms |

**Cost Optimization:** 85% of operations are deterministic (Layer 2), reducing LLM calls to synthesis only.

## Design Philosophy

Semantic Encoding Aesthetics operates on the principle that **meaning should shape form**. Rather than using prompts to tell an image generator what to create, this system extracts semantic content from text and maps it mathematically onto visual parameters. The result is:

- **Reproducible:** Same text + same parameters = same visual output
- **Interpretable:** Visual changes directly trace to semantic shifts
- **Composable:** Sentiment, encoding type, and palette are independent axes
- **Economical:** Deterministic operations eliminate redundant LLM calls

## Aesthetic Applications

- **Archival typography:** Preserve text as encoded visual artifact
- **Sentiment visualization:** Make emotional content visible and sonifiable
- **Data poetry:** Combine structured encoding with artistic composition
- **Accessibility-forward design:** Braille encoding as primary aesthetic, not afterthought
- **Temporal displacement:** Morse or dot-matrix encoding evokes historical eras
- **Multi-sensory art:** Synchronized visual + audio expressions of text

## References

- Morse code: [International Morse Code](https://en.wikipedia.org/wiki/Morse_code)
- Braille: [Unicode Braille Patterns](https://unicode.org/charts/PDF/U2800.pdf)
- Dot-matrix typography: [5×7 pixel font specifications](https://en.wikipedia.org/wiki/Dot_matrix_printing)
- Sentiment analysis: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Color extraction: k-means clustering on dominant image regions

## Examples in the Wild

- Text: "And they never heard surf music again"
  - Encoding: Dot Matrix
  - Sentiment: Neutral (factual finality)
  - Result: Crisp 37-character grid with clean, metallic dots against muted palette
  
- Text: "Our beloved life vests are melting!"
  - Encoding: Dot Matrix
  - Sentiment: Positive arousal + unpleasant valence
  - Result: 2.645x larger dots with 26.7% degradation, warm colors, visible dripping/dissolution

## Future Extensions

- **Dynamic text:** Animated encodings that evolve based on character-level sentiment shifts
- **Layered encoding:** Combine multiple encoding types in single composition
- **Haptic output:** Render Braille patterns for tactile devices
- **Spectral analysis:** Map audio frequency content to visual color/saturation
- **Machine learning palette generation:** Train model on sentiment + image → optimal palette

## License

MIT

## Contributing

Contributions welcome. Primary areas of interest:

- Sentiment analysis refinement (multi-lingual support, domain-specific lexicons)
- Encoding type expansion (binary, hexadecimal, symbolic systems)
- Color extraction improvements (perceptual uniformity, cultural color semantics)
- Composition pattern documentation (new use cases, workflow examples)

## Maintainer

Built as part of the Lushy aesthetic composition ecosystem.
