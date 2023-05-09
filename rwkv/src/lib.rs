// See https://github.com/saharNooby/rwkv.cpp/blob/master/rwkv/rwkv_cpp_model.py

use std::{
  cmp::Ordering,
  ffi::{c_char, CString},
  iter::Iterator,
  path::Path,
  ptr,
};

use rand::{distributions::WeightedIndex, prelude::Distribution};
use rwkv_sys;
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;

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

static END_OF_LINE_TOKEN: u32 = 187;

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
  model_binding: ModelBindings,
  tokenizer: Tokenizer,
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

    Ok(Model {
      model_binding: model_lib,
      tokenizer,
    })
  }

  pub fn create_session_custom<'a>(
    &'a self,
    session_options: &SessionOptions,
  ) -> Result<Session<'a>, RWKVError> {
    Session::new(&self, session_options)
  }

  pub fn create_session<'a>(&'a self) -> Result<Session<'a>, RWKVError> {
    self.create_session_custom(&Default::default())
  }
}

pub struct PredictResult {
  next_state: Vec<f32>,
  next_logits: Vec<f32>,
}

#[derive(Clone)]
pub struct Prompt {
  user: String,
  bot: String,
  separator: String,
  prompt: String,
}

impl Default for Prompt {
  fn default() -> Self {
    Prompt {
      user: "User".to_string(),
      bot: "Bot".to_string(),
      separator: ":".to_string(),
      prompt: "\nThe following is a verbose and detailed conversation between an AI assistant called Bot, and a human user called User. Bot is intelligent, knowledgeable, wise and polite.\n\nUser: french revolution what year\n\nBot: The French Revolution started in 1789, and lasted 10 years until 1799.\n\nUser: 3+5=?\n\nBot: The answer is 8.\n\nUser: Can you gues who I'll marry?\n\nBot: Only if you tell me more about yourself - what are your interests?\n\nUser: solve for a: 9-a=2\n\nBot: The answer is a = 7, because 9 - 7 = 2.\n\nUser: wat is lhc\n\nBot: LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.\n\n".to_string()
    }
  }
}

#[derive(Default)]
pub struct SessionOptions {
  prompt: Prompt,
}

pub struct Session<'a> {
  model: &'a Model,
  current_state: Option<Vec<f32>>,
  last_logits: Vec<f32>,
  initial_prompt: Prompt,
}

fn softmax(vec: &Vec<f32>) -> Vec<f32> {
  let denominator: f32 = vec.iter().map(|val| val.exp()).sum();

  vec.iter().map(|val| val.exp() / denominator).collect()
}

fn random_choice(items: &Vec<(usize, f32)>) -> usize {
  let mut rng = rand::thread_rng();

  let dist: WeightedIndex<f32> =
    WeightedIndex::new(items.iter().map(|(_id, p)| p)).expect("Invalid probabilities");

  let (id, _p) = items
    .get(dist.sample(&mut rng))
    .expect("Sample must exist in original Vec");

  *id
}

impl<'a> Session<'a> {
  pub fn new(model: &'a Model, options: &SessionOptions) -> Result<Session<'a>, RWKVError> {
    let mut chat = Session {
      model,
      current_state: None,
      last_logits: vec![],
      initial_prompt: options.prompt.clone(),
    };

    chat.consume_text(&chat.initial_prompt.prompt.clone())?;

    Ok(chat)
  }

  fn pick_token(&self, logits: &Vec<f32>, temp: f32, top_p: f32) -> usize {
    if temp <= f32::EPSILON {
      let (id, _val) = logits
        .iter()
        .enumerate()
        .filter(|(_id, p)| !f32::is_nan(**p) && f32::is_finite(**p))
        .max_by(|(_id1, p1), (_id2, p2)| p1.partial_cmp(p2).unwrap_or(Ordering::Equal))
        .expect("Max must be there");

      return id;
    }

    log::trace!("Logits size before filtering {}", logits.len());

    let mut probabilities = softmax(&logits)
      .into_iter()
      .enumerate()
      .filter(|(_id, p)| !f32::is_nan(*p) && f32::is_finite(*p))
      .collect::<Vec<_>>();

    log::trace!("Probabilities size before picking {}", probabilities.len());

    if top_p <= 1.0 - f32::EPSILON {
      probabilities.sort_by(|(_id1, p1), (_id2, p2)| p2.partial_cmp(p1).unwrap_or(Ordering::Equal));

      let mut running_sum: f32 = 0.0;
      probabilities = probabilities
        .into_iter()
        .take_while(|(_id, p)| {
          let take_more = running_sum < top_p;
          running_sum += p;
          take_more
        })
        .collect();
    }
    log::trace!(
      "Probabilities size after top_p {} = {:?}",
      probabilities.len(),
      probabilities
    );

    if temp <= 1.0 - f32::EPSILON {
      probabilities = probabilities
        .into_iter()
        .map(|(id, p)| (id, p.powf(1.0 / temp)))
        .collect();
    }

    log::trace!(
      "Probabilities size before random_choice {}",
      probabilities.len()
    );

    random_choice(&probabilities)
  }

  fn consume_single_token(&mut self, token: u32) -> Result<(), RWKVError> {
    let res = self
      .model
      .model_binding
      .predict(&self.current_state, token)?;
    self.last_logits = res.next_logits;
    self.current_state = Some(res.next_state);
    Ok(())
  }

  fn consume_text(&mut self, text: &str) -> Result<(), RWKVError> {
    let encoding = self
      .model
      .tokenizer
      .encode(text.to_string(), true)
      .map_err(|e| RWKVError::TokenEncodeError { source: e })?;

    log::debug!("Loading tokens...");
    for token in encoding.get_ids() {
      self.consume_single_token(*token)?;
    }

    Ok(())
  }

  fn produce_text(&mut self) -> Result<String, RWKVError> {
    let mut tokens: Vec<u32> = vec![];
    let mut max_tokens_per_response = 4096; // todo: configurable

    loop {
      let predicted_token = self.pick_token(&self.last_logits, 1.0, 0.5) as u32;

      if predicted_token == 0 {
        break;
      }
      if max_tokens_per_response <= 0 {
        break;
      }
      if predicted_token == END_OF_LINE_TOKEN && tokens.last().unwrap_or(&0) == &END_OF_LINE_TOKEN {
        break;
      } else {
        tokens.push(predicted_token);

        max_tokens_per_response -= 1;

        log::trace!("Predicted token: {}", predicted_token);
        self.consume_single_token(predicted_token)?;
      }
    }

    log::debug!("Predicted tokens {:?}", tokens);
    let res = self
      .model
      .tokenizer
      .decode(tokens, true)
      .map_err(|e| RWKVError::TokenDecodeError { source: e })?;

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
