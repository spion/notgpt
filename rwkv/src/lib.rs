// See https://github.com/saharNooby/rwkv.cpp/blob/master/rwkv/rwkv_cpp_model.py

use std::{
  ffi::{c_char, CString},
  path::Path,
  ptr,
};

use rwkv_sys;
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;

#[derive(Error, Debug)]
pub enum RWKVError {
  #[error("Tokens read failure")]
  TokenReadFailure { source: Box<dyn std::error::Error> },

  #[error("Tokens read failure")]
  ModelReadFailure { source: Box<dyn std::error::Error> },

  #[error("Tokens encoding error")]
  TokenEncodeError { source: Box<dyn std::error::Error> },

  #[error("Token prediction error")]
  TokenPredictionError,
}

pub struct RWKVModel {
  ctx: *mut rwkv_sys::rwkv_context,

  state_buffer_element_count: usize, //= self._library.rwkv_get_state_buffer_element_count(self._ctx)
  logits_buffer_element_count: usize, //= self._library.rwkv_get_logits_buffer_element_count(self._ctx)
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
        token as i32, // this should be OK - if not, hilarity ensues
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
}

//
// let tokenizer =
// Tokenizer::from_file(tokens_path).map_err(|e| RWKVError::TokenReadFailure { source: e })?;

// let encoding = self
// .tokenizer
// .encode(text, true)
// .map_err(|e| RWKVError::TokenEncodeError { source: e })?;

impl<'a> RWKVChat<'a> {
  pub fn new<P: AsRef<Path>>(model: &'a RWKVModel, tokenizer: &'a Tokenizer) -> RWKVChat<'a> {
    RWKVChat {
      model: model,
      tokenizer: tokenizer,
      current_state: None,
    }
  }
}
