// See https://github.com/saharNooby/rwkv.cpp/blob/master/rwkv/rwkv_cpp_model.py

use std::{
  ffi::{c_char, CString},
  path::Path,
  ptr,
  sync::Arc,
};

use rwkv_sys;
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;

mod token_functions;

#[derive(Error, Debug)]
pub enum RWKVError {
  #[error("Tokens read failure")]
  TokenReadFailure {
    #[source]
    source: Box<dyn std::error::Error + Send + Sync>,
  },

  #[error("Tokens read failure")]
  ModelReadFailure {
    #[source]
    source: Box<dyn std::error::Error + Send + Sync>,
  },

  #[error("Tokens encoding error")]
  TokenEncodeError {
    #[source]
    source: Box<dyn std::error::Error + Send + Sync>,
  },

  #[error("Tokens encoding error")]
  TokenDecodeError {
    #[source]
    source: Box<dyn std::error::Error + Send + Sync>,
  },

  #[error("Token prediction error")]
  TokenPredictionError,
}

struct ModelBindings {
  ctx: *mut rwkv_sys::rwkv_context,
  state_buffer_element_count: usize,
  logits_buffer_element_count: usize,
}

impl Drop for ModelBindings {
  fn drop(&mut self) {
    unsafe { rwkv_sys::rwkv_free(self.ctx) }
  }
}

// static END_OF_LINE_TOKEN: u32 = 187;

impl ModelBindings {
  pub fn new<P: AsRef<Path>>(model_path: &P, n_threads: u32) -> Result<ModelBindings, RWKVError> {
    // let test = model_path.into<String>();
    let model_str = model_path
      .as_ref()
      .to_str()
      .expect("Failed to convert Path to str");

    let cstring = CString::new(model_str).expect("Failed to create CString");
    let raw_ptr: *const c_char = cstring.as_ptr();

    let ctx = unsafe { rwkv_sys::rwkv_init_from_file(raw_ptr, n_threads) };

    let state_size = unsafe { rwkv_sys::rwkv_get_state_buffer_element_count(ctx) } as usize;
    let logits_size = unsafe { rwkv_sys::rwkv_get_logits_buffer_element_count(ctx) } as usize;

    Ok(ModelBindings {
      ctx: ctx,

      state_buffer_element_count: state_size,
      logits_buffer_element_count: logits_size,
    })
  }

  pub fn predict(
    &self,
    current_state: &Option<Vec<f32>>,
    token: u32,
  ) -> Result<PredictResult, RWKVError> {
    let mut next_logits: Vec<f32> = vec![0.0f32; self.logits_buffer_element_count];

    let mut next_state = match current_state {
      None => vec![0.0f32; self.state_buffer_element_count],
      Some(state) => state.clone(),
    };

    let state_arg = match current_state {
      None => ptr::null_mut(),
      Some(_) => next_state.as_mut_ptr(),
    };

    let success = unsafe {
      rwkv_sys::rwkv_eval(
        self.ctx,
        token,
        state_arg,
        next_state.as_mut_ptr(),
        next_logits.as_mut_ptr(),
      )
    };

    if success {
      Ok(PredictResult {
        next_state,
        next_logits,
      })
    } else {
      Err(RWKVError::TokenPredictionError)
    }
  }
}

pub struct Model {
  model_binding: Arc<ModelBindings>,
  tokenizer: Arc<Tokenizer>,
}

impl Model {
  pub fn new<P: AsRef<Path>>(
    model_path: &P,
    tokens_path: &P,
    n_threads: u32,
  ) -> Result<Model, RWKVError> {
    let model_lib = ModelBindings::new(model_path, n_threads)?;
    let tokenizer =
      Tokenizer::from_file(tokens_path).map_err(|e| RWKVError::TokenReadFailure { source: e })?;

    log::info!("Loaded model and tokenizer");
    Ok(Model {
      model_binding: Arc::new(model_lib),
      tokenizer: Arc::new(tokenizer),
    })
  }

  pub fn create_session_custom(
    &self,
    session_options: &SessionOptions,
  ) -> Result<Session, RWKVError> {
    Session::new(
      Arc::clone(&self.model_binding),
      Arc::clone(&self.tokenizer),
      session_options,
    )
  }

  pub fn create_session<'a>(&'a self) -> Result<Session, RWKVError> {
    self.create_session_custom(&Default::default())
  }
}

pub struct PredictResult {
  next_state: Vec<f32>,
  next_logits: Vec<f32>,
}

#[derive(Clone)]
pub struct Prompt {
  pub user: String,
  pub bot: String,
  pub separator: String,
  pub prompt: String,
}

impl Default for Prompt {
  fn default() -> Self {
    Prompt {
      user: "User".to_string(),
      bot: "Bot".to_string(),
      separator: ":".to_string(),
      prompt: r#"
The following is a verbose and detailed conversation between an AI assistant called Bot, and a human user called User. Bot is intelligent, knowledgeable, wise and polite.

User: french revolution what year

Bot: The French Revolution started in 1789, and lasted 10 years until 1799.

User: 3+5=?

Bot: The answer is 8.

User: Can you gues who I'll marry?

Bot: Only if you tell me more about yourself - what are your interests?

User: solve for a: 9-a=2

Bot: The answer is a = 7, because 9 - 7 = 2.

User: wat is lhc

Bot: LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

"#.to_string()
    }
  }
}

#[derive(Default)]
pub struct SessionOptions {
  pub prompt: Prompt,
}

pub struct Session {
  model: Arc<ModelBindings>,
  tokenizer: Arc<Tokenizer>,
  current_state: Option<Vec<f32>>,
  last_logits: Vec<f32>,
  initial_prompt: Prompt,
}

impl Session {
  fn new(
    model: Arc<ModelBindings>,
    tokenizer: Arc<Tokenizer>,
    options: &SessionOptions,
  ) -> Result<Session, RWKVError> {
    let mut chat = Session {
      model,
      tokenizer,
      current_state: None,
      last_logits: vec![],
      initial_prompt: options.prompt.clone(),
    };

    chat.consume_text(&chat.initial_prompt.prompt.clone())?;

    Ok(chat)
  }

  fn consume_single_token(&mut self, token: u32) -> Result<(), RWKVError> {
    let res = self.model.predict(&self.current_state, token)?;
    self.last_logits = res.next_logits;
    self.current_state = Some(res.next_state);
    Ok(())
  }

  pub fn consume_text(&mut self, text: &str) -> Result<(), RWKVError> {
    let encoding = self
      .tokenizer
      .encode(text.to_string(), true)
      .map_err(|e| RWKVError::TokenEncodeError { source: e })?;

    log::debug!("Loading {} = {} tokens...", text, encoding.get_ids().len());
    for token in encoding.get_ids() {
      self.consume_single_token(*token as u32)?;
    }

    Ok(())
  }

  pub fn produce_text(&mut self) -> Result<String, RWKVError> {
    let mut tokens: Vec<u32> = vec![];
    let mut max_tokens_per_response = 4096; // todo: configurable

    loop {
      if max_tokens_per_response % 8 == 0 {
        log::debug!("Generating tokens, {} left...", max_tokens_per_response);
      }

      let predicted_token =
        token_functions::sample_token(&self.last_logits, &tokens, Default::default()) as u32;

      if predicted_token == 0 {
        break;
      }
      if max_tokens_per_response <= 0 {
        break;
      }
      tokens.push(predicted_token);
      self.consume_single_token(predicted_token)?;
      max_tokens_per_response -= 1;

      let last_decoded = if tokens.len() > 1 {
        self
          .tokenizer
          .decode(tokens[tokens.len() - 2..tokens.len()].to_vec(), true)
          .map_err(|e| RWKVError::TokenDecodeError { source: e })?
      } else {
        "".to_string()
      };

      if last_decoded.contains("\n\n") {
        break;
      }
    }

    log::debug!("Predicted tokens {:?}", tokens);
    let res = self
      .tokenizer
      .decode(tokens, true)
      .map_err(|e| RWKVError::TokenDecodeError { source: e })?
      .trim()
      .to_string();

    Ok(res)
  }

  pub fn generate_response(&mut self, text: &str) -> Result<String, RWKVError> {
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
    let res = self.produce_text();
    log::info!("Produced response {:?}", res);
    res
  }
}
