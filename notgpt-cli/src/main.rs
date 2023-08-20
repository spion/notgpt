use anyhow::{anyhow, Context, Result};
use notgpt_model::{SessionManager, SessionOptions};
use rwkv::Model;
use std::io;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
  env_logger::init();

  log::info!("Loading model...");

  let model_path = std::env::var("MODEL_PATH")
    .with_context(|| "MODEL_PATH variable unset, set it to the path of the RWKV model")?;

  let tokens_path = std::env::var("TOKENS_PATH")
    .with_context(|| "TOKENS_PATH variable unset, set it to the path of the RWKV tokenizer json")?;

  let model = Model::new(&model_path, 6)
    .map_err(|e| anyhow!(e))
    .with_context(|| "Unable to initialize model")?;

  let tokenizer = Tokenizer::from_file(&tokens_path).unwrap();

  let mut sessionManager = SessionManager::new(model, tokenizer);

  let mut chat = sessionManager.create_session(&Default::default())?;

  log::info!("Sesison started");

  println!("> ");
  for line in io::stdin().lines() {
    let response = sessionManager.generate_response(&mut chat, &line?, 512)?;
    println!("{}", response);

    println!("> ")
  }
  Ok(())
}
