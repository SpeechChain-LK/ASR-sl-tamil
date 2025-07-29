# Automatic Speech Recognition (Tamil-Sinhala)

A comprehensive research project focused on building robust Automatic Speech Recognition (ASR) systems for Tamil language using OpenAI's Whisper model, with particular emphasis on noise-robust speech recognition.

## 🎯 Project Overview

This project develops and evaluates Tamil ASR systems using fine-tuned Whisper models, with extensive research on handling noisy audio conditions. The research encompasses data preprocessing, model training across multiple experiments, and comprehensive testing frameworks.

## 📊 Project Structure

```
├── 1. Research Papers/           # Literature review and research references
│   ├── other countries/         # International research papers
│   └── sri lanka/              # Regional research papers
├── 2. Data Pre-Processing/      # Data cleaning and preparation
├── 3. Tranning/                # Model training experiments
│   ├── EXP 01/                 # Clear audio training
│   ├── EXP 02/                 # Experiment 2
│   ├── EXP 03/                 # Experiment 3
│   ├── EXP 04/                 # Experiment 4
│   ├── EXP 05/                 # Noise-robust training
│   └── EXP 06/                 # Experiment 6
├── 4. Testing/                 # Model evaluation and testing
│   ├── Clear Audio/            # Clean audio test data
│   ├── Noisy Audio/           # Noisy audio test data
│   ├── kaggle testing/        # Kaggle platform testing
│   └── model/                 # Trained model files
└── README.md
```

## 🔬 Research Focus Areas

### 1. **Literature Review**
- Comprehensive analysis of Whisper model capabilities
- Research on low-resource language ASR systems
- Investigation of noise-robust speech recognition techniques
- Study of Tamil language ASR systems

### 2. **Data Preprocessing**
- Audio quality assessment using PESQ scores
- Noise detection and classification
- Data cleaning and validation
- Statistical analysis of audio features
- Gender-based data separation and analysis

### 3. **Model Training Experiments**
- **EXP 01**: Baseline Whisper fine-tuning on clear Tamil audio
- **EXP 05**: Advanced noise-robust ASR training with data augmentation
- Multiple experimental configurations with different hyperparameters
- Early stopping and model checkpointing strategies

### 4. **Testing and Evaluation**
- Comprehensive evaluation on both clear and noisy audio
- Word Error Rate (WER) analysis across different noise conditions
- Performance comparison across multiple speakers
- Real-world audio testing scenarios

## 🛠 Technical Implementation

### Base Model
- **Foundation**: OpenAI Whisper (small/base variants)
- **Language**: Tamil (ta)
- **Task**: Speech-to-text transcription

### Training Configuration
```python
# Key training parameters
- Batch Size: 16-48 (per device)
- Learning Rate: 1e-5 to 1.7e-5
- Epochs: 4-15 (with early stopping)
- Gradient Accumulation: 1-2 steps
- Optimizer: AdamW with weight decay
- Precision: FP16 for memory efficiency
```

### Data Processing Pipeline
1. **Audio Preprocessing**
   - Resampling to 16kHz
   - Mono channel conversion
   - Feature extraction using Whisper's processor

2. **Noise Robustness**
   - Multiple noise types: cafe, city sounds, traffic, etc.
   - Various SNR levels (0dB, 10dB, 20dB)
   - Data augmentation techniques

3. **Quality Assessment**
   - PESQ score evaluation
   - Statistical feature analysis
   - Confusion matrix generation

## 📈 Key Results

### Model Performance
- Successfully trained noise-robust Tamil ASR models
- Generated multiple model checkpoints for comparison
- Achieved competitive WER scores on both clean and noisy audio

### Data Insights
- Comprehensive analysis of 3000+ audio samples
- Gender-based performance variations documented
- Noise impact quantification across different conditions

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install transformers datasets evaluate
pip install librosa soundfile
pip install jiwer pandas matplotlib seaborn
pip install tensorboard
```

### Quick Usage
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load trained model
processor = WhisperProcessor.from_pretrained("path/to/model")
model = WhisperForConditionalGeneration.from_pretrained("path/to/model")

# Process audio
waveform, sr = torchaudio.load("tamil_audio.wav")
input_features = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features

# Generate transcription
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
```

## 📁 Key Files and Notebooks

### Data Preprocessing
- `2. Data Pre-Processing/data-preprocessing.ipynb` - Complete data analysis pipeline
- `cleaned_data.tsv` - Processed dataset
- Various visualization outputs for data quality assessment

### Training Notebooks
- `3. Tranning/EXP 01/whisper-clear-asr-train-EXP-01.ipynb` - Baseline training
- `3. Tranning/EXP 05/whisper-noisy-asr-train-EXP-05.ipynb` - Noise-robust training

### Testing Frameworks
- `4. Testing/WHISPER-TESTING-1.ipynb` - Comprehensive model evaluation
- `4. Testing/WHISPER-TESTING-2.ipynb` - Advanced testing scenarios
- `transcriptions_with_wer.tsv` - Detailed WER analysis results

## 🎯 Evaluation Metrics

- **WER (Word Error Rate)**: Primary metric for transcription accuracy
- **CER (Character Error Rate)**: Character-level accuracy assessment
- **PESQ Scores**: Audio quality measurement
- **SNR Analysis**: Signal-to-noise ratio impact evaluation

## 📊 Dataset Characteristics

- **Language**: Tamil
- **Speakers**: Multiple speakers (male/female)
- **Audio Quality**: Both clear and artificially noised samples
- **Noise Types**: Cafe, city sounds, traffic, telephone, etc.
- **Total Samples**: 3000+ audio files with transcriptions

## 🔮 Future Work

- Extension to Sinhala language ASR
- Real-time speech recognition implementation
- Mobile deployment optimization
- Cross-lingual speech translation capabilities
- Advanced noise reduction techniques

## 🤝 Contributing

This is a research project focused on Tamil ASR systems. For collaboration or questions about the research methodology, please refer to the detailed notebooks and experimental results.

## 📄 License

This project is developed for academic research purposes. Please cite appropriately if using any components or methodologies from this work.

## 📧 Contact

For research collaboration or technical questions, please refer to the institutional affiliations mentioned in the research papers.

---

*This project represents ongoing research in low-resource language ASR systems with a focus on noise robustness and practical deployment considerations.*
