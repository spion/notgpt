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

pub struct RWKVModel {
  ctx: *mut rwkv_sys::rwkv_context,
  state_buffer_element_count: usize,
  logits_buffer_element_count: usize,
}

impl Drop for RWKVModel {
  fn drop(&mut self) {
    unsafe { rwkv_sys::rwkv_free(self.ctx) }
  }
}

pub struct PredictResult {
  next_state: Vec<f32>,
  next_logits: Vec<f32>,
}

impl RWKVModel {
  pub fn new<P: AsRef<Path>>(model_path: &P, n_threads: u32) -> Result<RWKVModel, RWKVError> {
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

    Ok(RWKVModel {
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
    let mut next_state: Vec<f32> = vec![0.0f32; self.state_buffer_element_count];
    let mut next_logits: Vec<f32> = vec![0.0f32; self.logits_buffer_element_count];

    let state_arg = match current_state {
      None => ptr::null_mut(),
      Some(state) => state.clone().as_mut_ptr(),
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

pub struct RWKVChat<'a> {
  model: &'a RWKVModel,
  tokenizer: &'a Tokenizer,
  current_state: Option<Vec<f32>>,
  predicted_token: u32,
}

fn softmax(vec: &Vec<f32>) -> Vec<f32> {
  let denominator: f32 = vec.iter().map(|val| val.exp()).sum();

  vec.iter().map(|val| val.exp() / denominator).collect()
}

fn random_choice<T>(probabilities: T) -> usize
where
  T: Iterator<Item = f32>,
{
  let mut rng = rand::thread_rng();
  let dist = WeightedIndex::new(probabilities).expect("Invalid probabilities");

  dist.sample(&mut rng)
}

impl<'a> RWKVChat<'a> {
  pub fn new(model: &'a RWKVModel, tokenizer: &'a Tokenizer) -> RWKVChat<'a> {
    RWKVChat {
      model: model,
      tokenizer: tokenizer,
      current_state: None,
      predicted_token: 0,
    }
  }

  fn pick_token(&self, logits: Vec<f32>, temp: f32, top_p: f32) -> u32 {
    if temp <= f32::EPSILON {
      let (id, _val) = logits
        .iter()
        .enumerate()
        .max_by(|(_id1, p1), (_id2, p2)| p1.partial_cmp(p2).unwrap_or(Ordering::Equal))
        .expect("Max must be there");

      return id as u32;
    }
    let mut probabilities = softmax(&logits).into_iter().enumerate().collect::<Vec<_>>();

    if (top_p - 1.0).abs() > f32::EPSILON {
      probabilities.sort_by(|(_id1, p1), (_id2, p2)| p1.partial_cmp(p2).unwrap_or(Ordering::Equal));

      let mut running_sum: f32 = 0.0;
      probabilities = probabilities
        .into_iter()
        .take_while(|(_id, p)| {
          running_sum += p;
          running_sum < top_p || running_sum <= 0.0 // at least one
        })
        .collect();
    }

    if (temp - 1.0).abs() > f32::EPSILON {
      probabilities = probabilities
        .into_iter()
        .map(|(id, p)| (id, p.powf(temp)))
        .collect();
    }

    // let sum_probabilities: f32 = probabilities.iter().map(|(_id, p)| p).sum();

    random_choice(probabilities.into_iter().map(|(_id, p)| p)) as u32

    // let mut rng = rand::thread_rng();
    // let rand_probability: f32 = rng.gen();

    // let mut running_sum = 0.0;
    // let (stop_token_id, _) = probabilities
    //   .into_iter()
    //   .take_while(|(_id, p)| {
    //     running_sum += p / sum_probabilities;
    //     running_sum < rand_probability || running_sum <= 0.0 // at least one
    //   })
    //   .last()
    //   .expect("Unexpected zero tokens returned.");

    // stop_token_id
  }

  pub fn generate_response(&mut self, text: &str) -> Result<String, RWKVError> {
    let encoding = self
      .tokenizer
      .encode(text, true)
      .map_err(|e| RWKVError::TokenEncodeError { source: e })?;

    println!("Loading tokens...");
    for token in encoding.get_ids() {
      let res = self.model.predict(&self.current_state, *token)?;
      self.predicted_token = self.pick_token(res.next_logits, 0.0, 1.0) as u32;
      self.current_state = Some(res.next_state);
    }

    let mut tokens: Vec<u32> = vec![self.predicted_token];
    let mut max_tokens_per_response = 128; // todo: configurable

    println!(
      "Generating response... logits_size={}",
      self.model.logits_buffer_element_count
    );
    while self.predicted_token != 0 && max_tokens_per_response > 0 {
      let res = self
        .model
        .predict(&self.current_state, self.predicted_token)?;

      self.predicted_token = self.pick_token(res.next_logits, 0.0, 1.0) as u32;
      self.current_state = Some(res.next_state);

      if self.predicted_token != 0 {
        tokens.push(self.predicted_token)
      }

      max_tokens_per_response -= 1;
    }

    println!("Predicted tokens {:?}", tokens);
    let res = self
      .tokenizer
      .decode(tokens, true)
      .map_err(|e| RWKVError::TokenDecodeError { source: e })?;

    Ok(res)
  }

  // pub fn load_text(&mut self, text: &str)
}
