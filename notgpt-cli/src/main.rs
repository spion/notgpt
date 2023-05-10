use anyhow::{anyhow, Context, Result};
use rwkv::Model;
use std::io;

fn main() -> Result<()> {
  env_logger::init();

  log::info!("Loading model...");

  let model_path = std::env::var("MODEL_PATH")
    .with_context(|| "MODEL_PATH variable unset, set it to the path of the RWKV model")?;

  let tokens_path = std::env::var("TOKENS_PATH")
    .with_context(|| "TOKENS_PATH variable unset, set it to the path of the RWKV tokenizer json")?;

  let model = Model::new(&model_path, &tokens_path, 6)
    .map_err(|e| anyhow!(e))
    .with_context(|| "Unable ot initialize model")?;

  let mut chat = model.create_session()?;

  log::info!("Model loaded");

  println!("> ");
  for line in io::stdin().lines() {
    let response = chat.generate_response(&line?)?;
    println!("{}", response);

    println!("> ")
  }
  Ok(())
}
