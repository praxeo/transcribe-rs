use ndarray::{Array2, ArrayD};
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use std::path::Path;

use super::session;
use super::Quantization;
use crate::decode::{ctc_greedy_decode, load_vocab, CtcDecoderResult};
use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
    TranscriptionSegment,
};

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "MedASR",
    engine_id: "medasr",
    sample_rate: 16000,
    languages: &["en"],
    supports_timestamps: true,
    supports_translation: false,
    supports_streaming: false,
};

// LASR feature extractor parameters (from preprocessor_config.json)
const SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const NUM_MELS: usize = 128;
const F_MIN: f32 = 0.0;

#[derive(Debug, Clone, Default)]
pub struct MedAsrParams {
    pub language: Option<String>,
}

pub struct MedAsrModel {
    session: Session,
    vocab: Vec<String>,
    blank_idx: i64,
    mel_filters: Array2<f32>,
}

impl MedAsrModel {
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let model_path = session::resolve_model_path(model_dir, "medasr_int8_dynamic", quantization);
        let tokens_path = model_dir.join("tokens.txt");

        if !model_path.exists() {
            return Err(TranscribeError::ModelNotFound(model_path));
        }
        if !tokens_path.exists() {
            return Err(TranscribeError::ModelNotFound(tokens_path));
        }

        log::info!("Loading MedASR model from {:?}...", model_path);
        let session = session::create_session(&model_path)?;

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        log::info!("Model inputs: {:?}", input_names);

        let (vocab, blank_idx_from_vocab) = load_vocab(&tokens_path)?;
        let blank_idx = blank_idx_from_vocab.map(|v| v as i64).unwrap_or(0);
        log::info!("Loaded {} vocab tokens, blank_idx={}", vocab.len(), blank_idx);

        let mel_filters = build_mel_filters();

        Ok(Self { session, vocab, blank_idx, mel_filters })
    }

    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        _params: &MedAsrParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }

    fn infer(&mut self, samples: &[f32]) -> Result<TranscriptionResult, TranscribeError> {
        // 1. Compute LASR log-mel features → [num_frames, NUM_MELS]
        let mel = self.compute_lasr_features(samples);
        let num_frames = mel.shape()[0];
        log::debug!("Mel shape: {:?}, min={:.3} max={:.3}", mel.shape(),
            mel.iter().cloned().fold(f32::INFINITY, f32::min),
            mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        // 2. Reshape to [1, num_frames, NUM_MELS] contiguous
        let input_features = mel
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((1, num_frames, NUM_MELS))
            .map_err(|e| TranscribeError::Inference(format!("Reshape error: {}", e)))?;

        // 3. Attention mask: all-true, shape [1, num_frames]
        let attention_mask = Array2::<bool>::from_elem((1, num_frames), true);

        // 4. Run ONNX
        let logits: ArrayD<f32> = {
            let t_input = TensorRef::from_array_view(input_features.view())?;
            let t_mask = TensorRef::from_array_view(attention_mask.view())?;
            let outputs = self.session.run(inputs!["input_features" => t_input, "attention_mask" => t_mask])?;
            outputs
                .get("logits")
                .ok_or_else(|| TranscribeError::Inference("Missing output: logits".to_string()))?
                .try_extract_array::<f32>()?
                .to_owned()
        };

        log::debug!("Logits shape: {:?}", logits.shape());

        // 4. CTC greedy decode
        let num_logit_frames = logits.shape()[1];
        let logits_lengths = vec![num_logit_frames as i64];
        let logits_3d = logits
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| TranscribeError::Inference(format!("Logits dim error: {}", e)))?;
        let decoder_results = ctc_greedy_decode(&logits_3d.view(), &logits_lengths, self.blank_idx);

        Ok(self.convert_result(&decoder_results[0]))
    }

    /// LASR feature extraction matching Python LasrFeatureExtractor exactly:
    /// - Hann window of win_length=400, zero-padded to n_fft=512
    /// - Center padding: reflect-pad by n_fft/2 on each side
    /// - Power spectrum → mel filterbank → clamp(1e-5) → ln
    fn compute_lasr_features(&self, samples: &[f32]) -> Array2<f32> {
        let pad = N_FFT / 2; // 256 samples center padding

        // Reflect-pad the signal
        let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
        // Left pad: reflect first `pad` samples
        for i in (1..=pad).rev() {
            padded.push(if i < samples.len() { samples[i] } else { 0.0 });
        }
        padded.extend_from_slice(samples);
        // Right pad: reflect last `pad` samples
        let n = samples.len();
        for i in 1..=pad {
            padded.push(if n >= i + 1 { samples[n - 1 - i] } else { 0.0 });
        }

        let n_frames = if padded.len() >= WIN_LENGTH {
            (padded.len() - WIN_LENGTH) / HOP_LENGTH + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Array2::zeros((0, NUM_MELS));
        }

        // Hann window of length WIN_LENGTH (periodic: denominator = N, not N-1)
        let window: Vec<f32> = (0..WIN_LENGTH)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / WIN_LENGTH as f32).cos()))
            .collect();

        let freq_bins = N_FFT / 2 + 1;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);

        let mut features = Array2::<f32>::zeros((n_frames, NUM_MELS));

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;

            // Apply window to WIN_LENGTH samples, zero-pad to N_FFT
            let mut fft_buf = vec![Complex::new(0.0f32, 0.0f32); N_FFT];
            for i in 0..WIN_LENGTH {
                fft_buf[i] = Complex::new(padded[start + i] * window[i], 0.0);
            }

            fft.process(&mut fft_buf);

            // Power spectrum
            let power: Vec<f32> = fft_buf[..freq_bins].iter().map(|c| c.norm_sqr()).collect();

            // Mel filterbank + clamp + ln
            for m in 0..NUM_MELS {
                let energy: f32 = self.mel_filters.row(m)
                    .iter()
                    .zip(power.iter())
                    .map(|(&w, &p)| w * p)
                    .sum();
                features[[frame_idx, m]] = energy.max(1e-5_f32).ln();
            }
        }

        features
    }

    fn convert_result(&self, decoder_result: &CtcDecoderResult) -> TranscriptionResult {
        let frame_shift_s = HOP_LENGTH as f32 / SAMPLE_RATE as f32;

        // Skip special tokens: <epsilon>(0), <s>(1), </s>(2), <unk>(3)
        let text: String = decoder_result
            .tokens
            .iter()
            .filter(|&&id| id > 3)
            .filter_map(|&id| self.vocab.get(id as usize))
            .map(|t| t.replace('\u{2581}', " "))
            .collect::<Vec<_>>()
            .join("")
            .trim()
            .to_string();

        let segments: Option<Vec<TranscriptionSegment>> = if !decoder_result.timestamps.is_empty() {
            let segs: Vec<TranscriptionSegment> = decoder_result
                .tokens
                .iter()
                .zip(decoder_result.timestamps.iter())
                .enumerate()
                .filter_map(|(i, (&id, &t))| {
                    if id <= 3 { return None; }
                    let token_text = self.vocab.get(id as usize)
                        .map(|s| s.replace('\u{2581}', " "))?;
                    if token_text.trim().is_empty() { return None; }
                    let start = t as f32 * frame_shift_s;
                    let end = decoder_result.timestamps.get(i + 1)
                        .map(|&next| next as f32 * frame_shift_s)
                        .unwrap_or(start + frame_shift_s);
                    Some(TranscriptionSegment { start, end, text: token_text })
                })
                .collect();
            if segs.is_empty() { None } else { Some(segs) }
        } else {
            None
        };

        TranscriptionResult { text, segments }
    }
}

impl SpeechModel for MedAsrModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }
}

/// Build mel filterbank matrix [NUM_MELS, freq_bins] matching librosa/torchaudio defaults.
/// Uses HTK mel scale, freq_bins = N_FFT/2 + 1.
fn build_mel_filters() -> Array2<f32> {
    let freq_bins = N_FFT / 2 + 1;
    let f_max = SAMPLE_RATE as f32 / 2.0;

    let mel_min = hz_to_mel(F_MIN);
    let mel_max = hz_to_mel(f_max);

    // NUM_MELS + 2 evenly spaced mel points
    let mel_points: Vec<f32> = (0..=NUM_MELS + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (NUM_MELS + 1) as f32)
        .collect();

    // Convert to FFT bin indices
    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| mel_to_hz(m) / f_max * (freq_bins - 1) as f32)
        .collect();

    let mut filters = Array2::<f32>::zeros((NUM_MELS, freq_bins));

    for m in 0..NUM_MELS {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..freq_bins {
            let kf = k as f32;
            if kf > left && kf <= center {
                filters[[m, k]] = (kf - left) / (center - left);
            } else if kf > center && kf < right {
                filters[[m, k]] = (right - kf) / (right - center);
            }
        }
    }

    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}
