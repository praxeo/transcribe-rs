use ndarray::{Array2, ArrayD};
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
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

/// Per-model inference parameters for MedASR.
#[derive(Debug, Clone, Default)]
pub struct MedAsrParams {
    /// Language hint (currently unused, MedASR is English-only).
    pub language: Option<String>,
}

pub struct MedAsrModel {
    session: Session,
    vocab: Vec<String>,
    blank_idx: i64,
    input_names: Vec<String>,
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
        log::debug!("Model inputs: {:?}", input_names);

        let (vocab, blank_idx_from_vocab) = load_vocab(&tokens_path)?;
        let vocab_size = vocab.len();

        // MedASR uses blank token 0 by convention, but honor vocab metadata if present.
        let blank_idx = blank_idx_from_vocab.map(|v| v as i64).unwrap_or(0);

        log::info!(
            "Loaded MedASR vocabulary with {} tokens, blank_idx={}",
            vocab_size,
            blank_idx
        );

        Ok(Self {
            session,
            vocab,
            blank_idx,
            input_names,
        })
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        _params: &MedAsrParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }

    fn infer(&mut self, samples: &[f32]) -> Result<TranscriptionResult, TranscribeError> {
        log::debug!(
            "Transcribing {} samples ({:.2}s)",
            samples.len(),
            samples.len() as f32 / CAPABILITIES.sample_rate as f32
        );

        // 1. Normalize the raw waveform
        let normalized = self.normalize_waveform(samples);

        // 2. Prepare input tensor [1, num_samples]
        let audio = Array2::from_shape_vec((1, normalized.len()), normalized)?;
        let audio_dyn = audio.into_dyn();

        // 3. Run ONNX forward pass
        let logits = self.forward(&audio_dyn)?;

        log::debug!("Logits shape: {:?}", logits.shape());

        // 4. CTC greedy decode
        // ctc_greedy_decode expects a fixed ArrayView3 — convert from ArrayD
        let num_frames = logits.shape()[1];
        let logits_lengths = vec![num_frames as i64];
        let logits_3d = logits
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| TranscribeError::Inference(format!("Logits shape error: {}", e)))?;
        let decoder_results = ctc_greedy_decode(&logits_3d.view(), &logits_lengths, self.blank_idx);

        // 5. Convert result
        let result = self.convert_result(&decoder_results[0]);
        Ok(result)
    }

    fn normalize_waveform(&self, samples: &[f32]) -> Vec<f32> {
        let max_abs = samples.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
        if max_abs > 0.0 {
            samples.iter().map(|&x| x / max_abs).collect()
        } else {
            samples.to_vec()
        }
    }

    fn forward(&mut self, audio: &ArrayD<f32>) -> Result<ArrayD<f32>, TranscribeError> {
        let t_input_values = TensorRef::from_array_view(audio.view())?;

        // MedASR Wav2Vec2 model typically uses "input_values" as input name
        let input_name = if self.input_names.contains(&"input_values".to_string()) {
            "input_values"
        } else if self.input_names.contains(&"input".to_string()) {
            "input"
        } else {
            &self.input_names[0]
        };

        let inputs = inputs![
            input_name => t_input_values,
        ];

        let outputs = self.session.run(inputs)?;
        let logits = outputs
            .get("logits")
            .ok_or_else(|| TranscribeError::Inference("Missing output: logits".to_string()))?
            .try_extract_array::<f32>()?;

        Ok(logits.to_owned())
    }

    fn convert_result(
        &self,
        decoder_result: &CtcDecoderResult,
    ) -> TranscriptionResult {
        let tokens = &decoder_result.tokens;
        let timestamps = &decoder_result.timestamps;

        // Build text from token IDs
        let text: String = tokens
            .iter()
            .filter_map(|&id| {
                let idx = id as usize;
                if idx < self.vocab.len() {
                    let token = &self.vocab[idx];
                    // Replace sentencepiece underscore with space
                    Some(token.replace('\u{2581}', " "))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
            .trim()
            .to_string();

        // Calculate timestamps in seconds
        // Wav2Vec2 typically downsamples by ~320x (16000 Hz -> 50 Hz)
        let subsampling_factor = 320.0;
        let frame_shift_s = subsampling_factor / CAPABILITIES.sample_rate as f32;

        let segments = if !timestamps.is_empty() {
            let mut segs = Vec::new();
            for (i, &t) in timestamps.iter().enumerate() {
                let start_time = t as f32 * frame_shift_s;
                let end_time = timestamps
                    .get(i + 1)
                    .map(|&next| next as f32 * frame_shift_s)
                    .unwrap_or(start_time + 0.02);
                
                let token_text = tokens
                    .get(i)
                    .and_then(|&id| self.vocab.get(id as usize))
                    .map(|t| t.replace('\u{2581}', " "))
                    .unwrap_or_default();

                if !token_text.trim().is_empty() {
                    segs.push(TranscriptionSegment {
                        start: start_time,
                        end: end_time,
                        text: token_text,
                    });
                }
            }
            if segs.is_empty() {
                None
            } else {
                Some(segs)
            }
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
