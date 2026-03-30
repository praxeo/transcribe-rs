use std::path::PathBuf;
use transcribe_rs::onnx::medasr::{MedAsrModel, MedAsrParams};
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Path to the MedASR model directory
    let model_dir = PathBuf::from("models/medasr");
    
    // Load the model
    let mut model = MedAsrModel::load(&model_dir, &Quantization::Int8)?;
    
    println!("MedASR model loaded successfully!");
    println!("Capabilities: {:?}", model.capabilities());
    
    // Example: transcribe a WAV file
    let audio_path = PathBuf::from("test.wav");
    if audio_path.exists() {
        let result = model.transcribe_file(&audio_path, &Default::default())?;
        println!("\nTranscription: {}", result.text);
        if let Some(segments) = result.segments {
            println!("\nSegments:");
            for seg in segments {
                println!("  [{:.2}s - {:.2}s] {}", seg.start, seg.end, seg.text);
            }
        }
    } else {
        println!("No test.wav found. Please provide a 16kHz mono WAV file.");
    }
    
    Ok(())
}
