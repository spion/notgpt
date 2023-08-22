use anyhow::Result;

pub struct SessionOptions {
  initial_prompt: String,
}

pub trait Session {
  fn consume_text(&mut self, text: String) -> Result<()>;

  fn produce_text(&mut self) -> Result<String>;
}

pub trait Model {
  fn create_session(&self, options: SessionOptions) -> Result<Box<dyn Session>>;
}
