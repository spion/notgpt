use anyhow::Result;

pub mod token_functions;

pub trait TokenMidldeware {
  fn modify_probabilities(
    &mut self,
    model: dyn Model,
    session: dyn Session,
    token_probabilities: &mut Vec<u32>,
  );
}

pub struct SampleOptions {
  pub temp: f32,
  pub top_p: f32,
  pub repeat_penalty: f32,
  pub repeat_len: usize,
  pub middleware: Option<Box<dyn TokenMidldeware>>,
}

pub trait Session {
  fn consume_text(&mut self, text: String) -> Result<()>;

  fn produce_text(&mut self, opts: &SampleOptions) -> Result<String>;

  fn get_history(&self) -> String;
}

pub trait Model {
  fn create_session(&self) -> Result<Box<dyn Session>>;

  fn tokens2string(&self, tokens: &Vec<u32>) -> String;
  fn string2tokens(&self, text: &str) -> Vec<u32>;
}

#[cfg(test)]
mod tests {
  // use super::*;

  // #[test]
  // fn it_works() {
  //   let result = add(2, 2);
  //   assert_eq!(result, 4);
  // }
}
