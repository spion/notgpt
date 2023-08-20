use std::collections::HashMap;

mod token_functions;

use notgpt_model_interface::{GenericModel, NotGptError};
use tokenizers::Tokenizer;

use crate::token_functions::sample_token;

#[derive(Clone)]
pub struct Prompt {
  pub user: String,
  pub bot: String,
  pub separator: String,
  pub prompt: String,
  // pub beginning_token: Option<usize>,
  // pub ending_token: usize,
}

impl Default for Prompt {
  fn default() -> Self {
    let bot_name = "Assistant";
    Prompt {
      user: "User".to_string(),
      bot: bot_name.to_string(),
      separator: ":".to_string(),
      prompt: format!(
        r#"
The following is a verbose and detailed conversation between an AI assistant and a human user. The assistant is intelligent, knowledgeable, wise and polite.

"#
      ),
    }
  }
}

#[derive(Default)]
pub struct SessionOptions {
  pub prompt: Prompt,
}

pub struct Session<State> {
  state: State,
  text_so_far: String,
  last_logits: Vec<f32>,
  initial_prompt: Prompt,
}
pub struct SessionManager<Model: GenericModel> {
  model: Model,
  tokenizer: Tokenizer,
  sessions: HashMap<String, Session<Model::SessionState>>,
}

impl<Model: GenericModel> SessionManager<Model> {
  pub fn new(model: Model, tokenizer: Tokenizer) -> SessionManager<Model> {
    SessionManager {
      model,
      tokenizer,
      sessions: HashMap::new(),
    }
  }

  pub fn create_session(
    &mut self,
    options: &SessionOptions,
  ) -> Result<Session<Model::SessionState>, NotGptError> {
    let mut chat: Session<Model::SessionState> = Session {
      state: Default::default(),
      text_so_far: "".to_string(),
      last_logits: vec![],
      initial_prompt: options.prompt.clone(),
    };

    let prompt = chat.initial_prompt.prompt.clone();

    self.consume_text(&mut chat, &prompt)?;

    Ok(chat)
  }

  // fn with_session(id: &str) {}

  fn consume_text(
    &mut self,
    chat: &mut Session<Model::SessionState>,
    text: &str,
  ) -> Result<(), NotGptError> {
    let encoding = self
      .tokenizer
      .encode(text.to_string(), true)
      .map_err(|e| NotGptError::TokenEncodeError { source: e })?;

    log::debug!(
      "Loading '{}' = {} tokens...",
      text,
      encoding.get_ids().len()
    );

    chat.text_so_far += text;

    self.model.load_session_state(&mut chat.state);

    for token in encoding.get_ids() {
      self.consume_single_token(chat, *token as u32)?;
    }

    self.model.save_session_state(&mut chat.state);

    Ok(())
  }

  fn consume_single_token(
    &mut self,
    chat: &mut Session<Model::SessionState>,
    token: u32,
  ) -> Result<(), NotGptError> {
    let input_tokens = vec![token];
    let res = self.model.predict_logits(&mut chat.state, &input_tokens)?;
    chat.last_logits = res;
    Ok(())
  }

  fn produce_text(
    &mut self,
    chat: &mut Session<Model::SessionState>,
    max_tokens: u32,
  ) -> Result<String, NotGptError> {
    let mut tokens: Vec<u32> = vec![];
    let mut max_tokens_per_response = max_tokens;

    self.model.load_session_state(&mut chat.state);

    loop {
      if max_tokens_per_response % 8 == 0 {
        log::debug!("Generating tokens, {} left...", max_tokens_per_response);
      }

      let predicted_token = sample_token(&chat.last_logits, &tokens, Default::default()) as u32;

      if predicted_token == Model::STOP_TOKEN {
        break;
      }
      if max_tokens_per_response <= 0 {
        break;
      }

      tokens.push(predicted_token);
      self.consume_single_token(chat, predicted_token)?;
      max_tokens_per_response -= 1;

      let last_decoded = if tokens.len() > 2 {
        self
          .tokenizer
          .decode(tokens[tokens.len() - 3..tokens.len()].to_vec(), true)
          .map_err(|e| NotGptError::TokenDecodeError { source: e })?
      } else {
        "".to_string()
      };

      if last_decoded.contains("\n\n\n") {
        break;
      }
    }

    self.model.save_session_state(&mut chat.state);

    log::debug!("Predicted tokens {:?}", tokens);
    let res = self
      .tokenizer
      .decode(tokens, true)
      .map_err(|e| NotGptError::TokenDecodeError { source: e })?;

    chat.text_so_far += &format!("[[{}]]", res);

    // log::debug!("Text so far: {}", self.text_so_far);
    Ok(res.trim().to_string())
  }

  pub fn generate_response(
    &mut self,
    chat: &mut Session<Model::SessionState>,
    text: &str,
    max_tokens: u32,
  ) -> Result<String, NotGptError> {
    let input_format = format!(
      "{}{} {}\n\n{}{}",
      chat.initial_prompt.user,
      chat.initial_prompt.separator,
      text,
      chat.initial_prompt.bot,
      chat.initial_prompt.separator
    );

    log::info!("Patterning response with {:?}", input_format);
    self.consume_text(chat, &input_format)?;
    let res = self.produce_text(chat, max_tokens);
    log::info!("Produced response {:?}", res);

    res
  }
}
