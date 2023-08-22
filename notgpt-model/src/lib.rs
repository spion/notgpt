use std::{
  collections::{hash_map::Entry, HashMap},
  hash::Hash,
  path::Path,
  sync::{Arc, Mutex},
};

mod token_functions;

use keepcalm::SharedMut;
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

pub struct Session<Model: GenericModel> {
  model: Arc<Mutex<Model>>,
  tokenizer: Arc<Tokenizer>,
  state: Model::SessionState,
  text_so_far: String,
  last_logits: Vec<f32>,
  initial_prompt: Prompt,
}

impl<Model: GenericModel> Session<Model> {
  pub fn new(
    model: Arc<Mutex<Model>>,
    tokenizer: Arc<Tokenizer>,
    options: &SessionOptions,
  ) -> Result<Session<Model>, NotGptError> {
    let mut res = Session {
      model: model,
      tokenizer: tokenizer,
      state: Default::default(),
      text_so_far: "".to_string(),
      last_logits: vec![],
      initial_prompt: options.prompt.clone(),
    };

    res.consume_text(&options.prompt.prompt)?;

    Ok(res)
  }

  fn consume_text(&mut self, text: &str) -> Result<(), NotGptError> {
    let encoding = self
      .tokenizer
      .encode(text.to_string(), true)
      .map_err(|e| NotGptError::TokenEncodeError { source: e })?;

    log::debug!(
      "Loading '{}' = {} tokens...",
      text,
      encoding.get_ids().len()
    );

    let sm = self.model.clone();

    let mut model = sm
      .lock()
      .map_err(|_| NotGptError::IntertwinedModelUsageError)?;

    model.load_session_state(&mut self.state);

    for token in encoding.get_ids() {
      self.consume_single_token(&mut model, *token as u32)?;
    }

    model.save_session_state(&mut self.state);

    self.text_so_far += text;
    Ok(())
  }

  fn consume_single_token(&mut self, model: &mut Model, token: u32) -> Result<(), NotGptError> {
    let input_tokens = vec![token];
    let res = model.predict_logits(&mut self.state, &input_tokens)?;
    self.last_logits = res;
    Ok(())
  }

  fn produce_text(&mut self, max_tokens: u32) -> Result<String, NotGptError> {
    let mut tokens: Vec<u32> = vec![];
    let mut max_tokens_per_response = max_tokens;

    let sm = self.model.clone();

    let mut model = sm
      .lock()
      .map_err(|_| NotGptError::IntertwinedModelUsageError)?;

    model.load_session_state(&mut self.state);

    loop {
      if max_tokens_per_response % 8 == 0 {
        log::debug!("Generating tokens, {} left...", max_tokens_per_response);
      }

      let predicted_token = sample_token(&self.last_logits, &tokens, Default::default()) as u32;

      if predicted_token == Model::STOP_TOKEN {
        break;
      }
      if max_tokens_per_response <= 0 {
        break;
      }

      tokens.push(predicted_token);
      self.consume_single_token(&mut model, predicted_token)?;
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

    model.save_session_state(&mut self.state);

    log::debug!("Predicted tokens {:?}", tokens);
    let res = self
      .tokenizer
      .decode(tokens, true)
      .map_err(|e| NotGptError::TokenDecodeError { source: e })?;

    self.text_so_far += &format!("[[{}]]", res);

    // log::debug!("Text so far: {}", self.text_so_far);
    Ok(res.trim().to_string())
  }
}

pub trait AnySession {
  fn generate_response(&mut self, text: &str, max_tokens: u32) -> Result<String, NotGptError>;
  fn produce_text(&mut self, max_tokens: u32) -> Result<String, NotGptError>;
  fn consume_text(&mut self, text: &str) -> Result<(), NotGptError>;
}

impl<M: GenericModel> AnySession for Session<M> {
  fn generate_response(&mut self, text: &str, max_tokens: u32) -> Result<String, NotGptError> {
    let input_format = format!(
      "{}{} {}\n\n{}{}",
      self.initial_prompt.user,
      self.initial_prompt.separator,
      text,
      self.initial_prompt.bot,
      self.initial_prompt.separator
    );

    log::info!("Patterning response with {:?}", input_format);
    self.consume_text(&input_format)?;
    let res = self.produce_text(max_tokens);
    log::info!("Produced response {:?}", res);

    res
  }

  fn produce_text(&mut self, max_tokens: u32) -> Result<String, NotGptError> {
    self.produce_text(max_tokens)
  }

  fn consume_text(&mut self, text: &str) -> Result<(), NotGptError> {
    self.consume_text(text)
  }
}

pub struct SessionManager<Model: GenericModel> {
  model: Arc<Mutex<Model>>,
  tokenizer: Arc<Tokenizer>,
  sessions: HashMap<String, Session<Model>>,
  session_options: SessionOptions,
}

pub fn new_tokenizer<P: AsRef<Path>>(tokenizer_path: P) -> Result<Tokenizer, NotGptError> {
  return Tokenizer::from_file(tokenizer_path).map_err(|_| NotGptError::TokenPredictionError);
}

impl<Model: GenericModel> SessionManager<Model> {
  pub fn new<P: AsRef<Path>>(
    model: Model,
    tokenizer_path: P,
    options: SessionOptions,
  ) -> Result<SessionManager<Model>, NotGptError> {
    Ok(SessionManager {
      model: Arc::new(Mutex::new(model)),
      tokenizer: Arc::new(new_tokenizer(tokenizer_path)?),
      sessions: HashMap::new(),
      session_options: options,
    })
  }
}

pub trait AnySessionManager {
  fn get_session(&mut self, session_id: &String) -> Result<&mut dyn AnySession, NotGptError>;
}

impl<Model: GenericModel> AnySessionManager for SessionManager<Model> {
  fn get_session(&mut self, id: &String) -> Result<&mut dyn AnySession, NotGptError> {
    match self.sessions.entry(id.to_string()) {
      Entry::Occupied(occupied) => {
        // If the entry exists, return a reference to the existing ChatbotSession
        let res = occupied.into_mut();
        Ok(res)
      }
      Entry::Vacant(vacant) => {
        let session = Session::new(
          self.model.clone(),
          self.tokenizer.clone(),
          &self.session_options,
        )?;

        let res = vacant.insert(session);

        Ok(res)
      }
    }
  }
}
