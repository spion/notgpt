use std::io;

use rwkv::{RWKVChat, RWKVModel};

use anyhow::{anyhow, Context, Result};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
  println!("Loading model...");

  let model_path = std::env::var("MODEL_PATH")
    .with_context(|| "MODEL_PATH variable unset, set it to the path of the RWKV model")?;

  let tokens_path = std::env::var("TOKENS_PATH")
    .with_context(|| "TOKENS_PATH variable unset, set it to the path of the RWKV tokenizer json")?;

  let model = RWKVModel::new(&model_path, 6)
    .map_err(|e| anyhow!(e))
    .with_context(|| "Unable ot initialize model")?;

  let tokenizer = Tokenizer::from_file(tokens_path)
    .map_err(|e| anyhow!(e))
    .with_context(|| "Could not initialize tokenizer")?;

  let mut chat = RWKVChat::new(&model, &tokenizer);

  println!("Model loaded");
  print!("> ");
  for line in io::stdin().lines() {
    let response = chat.generate_response(&line?)?;
    println!("{}", response);

    print!("> ")
  }
  Ok(())
}
