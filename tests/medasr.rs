mod common;

use std::path::PathBuf;

use transcribe_rs::onnx::medasr::{MedAsrModel, MedAsrParams};
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

#[test]
fn test_medasr_transcription() {
    let model_path = PathBuf::from("models/medasr");
    let audio_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_path, &audio_path]) {
        return;
    }

    let mut model = MedAsrModel::load(&model_path, &Quantization::Int8).expect("Failed to load model");

    let result = model
        .transcribe_file(&audio_path, &transcribe_rs::TranscribeOptions::default())
        .expect("Failed to transcribe");

    // MedASR is optimized for medical speech but should still handle general English
    // We'll just check that it produces some output
    assert!(!result.text.is_empty(), "Transcription should not be empty");
    
    println!("MedASR transcription: {}", result.text);
}

#[test]
fn test_timestamps() {
    let model_path = PathBuf::from("models/medasr");
    let audio_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_path, &audio_path]) {
        return;
    }

    let mut model = MedAsrModel::load(&model_path, &Quantization::Int8).expect("Failed to load model");

    let result = model
        .transcribe_file(&audio_path, &transcribe_rs::TranscribeOptions::default())
        .expect("Failed to transcribe");

    assert!(
        result.segments.is_some(),
        "Transcription should return segments"
    );

    let segments = result.segments.unwrap();
    assert!(!segments.is_empty(), "Segments should not be empty");

    // Verify segment properties
    for (i, segment) in segments.iter().enumerate() {
        assert!(
            segment.start >= 0.0,
            "Segment {} start time should be non-negative, got {}",
            i,
            segment.start
        );

        assert!(
            segment.end >= segment.start,
            "Segment {} end time ({}) should be >= start time ({})",
            i,
            segment.end,
            segment.start
        );

        assert!(
            !segment.text.is_empty(),
            "Segment {} should have non-empty text",
            i
        );
    }

    // Verify chronological order
    for i in 1..segments.len() {
        assert!(
            segments[i].start >= segments[i - 1].start,
            "Segments should be in chronological order: segment {} starts at {} but segment {} starts at {}",
            i,
            segments[i].start,
            i - 1,
            segments[i - 1].start
        );
    }
}

#[test]
fn test_capabilities() {
    let model_path = PathBuf::from("models/medasr");

    if !common::require_paths(&[&model_path]) {
        return;
    }

    let model = MedAsrModel::load(&model_path, &Quantization::Int8).expect("Failed to load model");
    let caps = model.capabilities();

    assert_eq!(caps.name, "MedASR");
    assert_eq!(caps.engine_id, "medasr");
    assert_eq!(caps.sample_rate, 16000);
    assert!(caps.languages.contains(&"en"));
    assert!(caps.supports_timestamps);
    assert!(!caps.supports_translation);
    assert!(!caps.supports_streaming);
}
