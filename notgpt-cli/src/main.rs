use anyhow::{anyhow, Context, Result};
use notgpt_model::{AnySession, Session};
use rwkv::Model;
use std::{
  io,
  sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
  env_logger::init();

  log::info!("Loading model...");

  let model_path = std::env::var("MODEL_PATH")
    .with_context(|| "MODEL_PATH variable unset, set it to the path of the RWKV model")?;

  let tokens_path = std::env::var("TOKENS_PATH")
    .with_context(|| "TOKENS_PATH variable unset, set it to the path of the RWKV tokenizer json")?;

  let model = Arc::new(Mutex::new(
    Model::new(&model_path, 6)
      .map_err(|e| anyhow!(e))
      .with_context(|| "Unable to initialize model")?,
  ));

  let tokenizer = Arc::new(Tokenizer::from_file(&tokens_path).unwrap());

  let mut chat = Session::new(model.clone(), tokenizer.clone(), &Default::default())?;

  log::info!("Sesison started");

  println!("> ");
  for line in io::stdin().lines() {
    let response = chat.generate_response(&line?, 512)?;
    println!("{}", response);

    println!("> ")
  }
  Ok(())
}
