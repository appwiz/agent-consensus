use std::fs;
use std::io::{self, Read};
use std::process;

use clap::Parser;

use agent_consensus::engine::ConsensusEngine;
use agent_consensus::models::ConsensusInput;
use agent_consensus::strategies::STRATEGY_NAMES;

#[derive(Parser)]
#[command(name = "consensus", about = "Build consensus from multiple agent responses.")]
struct Cli {
    /// Path to input JSON file (default: stdin)
    #[arg(short, long)]
    input: Option<String>,

    /// Consensus strategy to use
    #[arg(short, long, default_value = "quorum")]
    strategy: String,

    /// Similarity threshold 0.0-1.0
    #[arg(short, long, default_value = "0.6")]
    threshold: f64,

    /// Pretty-print JSON output
    #[arg(short, long)]
    pretty: bool,

    /// List available strategies and exit
    #[arg(long)]
    list_strategies: bool,
}

fn main() {
    let cli = Cli::parse();

    if cli.list_strategies {
        let names: Vec<&str> = STRATEGY_NAMES.to_vec();
        println!("{}", serde_json::to_string(&names).unwrap());
        process::exit(0);
    }

    // Read input
    let raw = match &cli.input {
        Some(path) => match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading input: {e}");
                process::exit(1);
            }
        },
        None => {
            let mut buf = String::new();
            if let Err(e) = io::stdin().read_to_string(&mut buf) {
                eprintln!("Error reading input: {e}");
                process::exit(1);
            }
            buf
        }
    };

    let input_data: ConsensusInput = match serde_json::from_str(&raw) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading input: {e}");
            process::exit(1);
        }
    };

    // Run consensus
    let engine = match ConsensusEngine::new(&cli.strategy, cli.threshold) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    };

    let result = match engine.run(&input_data) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    };

    // Output
    let json = if cli.pretty {
        serde_json::to_string_pretty(&result).unwrap()
    } else {
        serde_json::to_string(&result).unwrap()
    };
    println!("{json}");

    if !result.consensus_reached {
        process::exit(2);
    }
}
