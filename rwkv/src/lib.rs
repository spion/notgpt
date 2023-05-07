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
}

pub struct RWKVModel {
  ctx: *mut rwkv_sys::rwkv_context,
  current_state: Option<Vec<f32>>,
  tokenizer: Tokenizer,

  state_buffer_element_count: usize, //= self._library.rwkv_get_state_buffer_element_count(self._ctx)
  logits_buffer_element_count: usize, //= self._library.rwkv_get_logits_buffer_element_count(self._ctx)
}

impl Drop for RWKVModel {
  fn drop(&mut self) {
    unsafe { rwkv_sys::rwkv_free(self.ctx) }
  }
}

impl RWKVModel {
  pub fn new<P: AsRef<Path>>(
    model_path: &P,
    tokens_path: &P,
    n_threads: u32,
  ) -> Result<RWKVModel, RWKVError> {
    // let test = model_path.into<String>();
    let model_str = model_path
      .as_ref()
      .to_str()
      .expect("Failed to convert Path to str");

    let cstring = CString::new(model_str).expect("Failed to create CString");
    let raw_ptr: *const c_char = cstring.as_ptr();

    let ctx = unsafe { rwkv_sys::rwkv_init_from_file(raw_ptr, n_threads) };

    let tokenizer =
      Tokenizer::from_file(tokens_path).map_err(|e| RWKVError::TokenReadFailure { source: e })?;

    let state_size = unsafe { rwkv_sys::rwkv_get_state_buffer_element_count(ctx) } as usize;
    let logits_size = unsafe { rwkv_sys::rwkv_get_logits_buffer_element_count(ctx) } as usize;

    Ok(RWKVModel {
      ctx: ctx,
      tokenizer: tokenizer,
      current_state: None,
      state_buffer_element_count: state_size,
      logits_buffer_element_count: logits_size,
    })
  }

  pub fn predict(self, text: &str) -> Result<&str, RWKVError> {
    let encoding = self
      .tokenizer
      .encode(text, true)
      .map_err(|e| RWKVError::TokenEncodeError { source: e })?;

    let mut next_state: Vec<f32> = vec![0.0f32; self.state_buffer_element_count];
    let mut next_logits: Vec<f32> = vec![0.0f32; self.logits_buffer_element_count];

    for token in encoding.get_ids() {
      let state_arg = match &self.current_state {
        None => ptr::null_mut(),
        Some(state) => state.clone().as_mut_ptr(),
      };

      let res = unsafe {
        rwkv_sys::rwkv_eval(
          self.ctx,
          *token as i32, // this should be OK - if not, hilarity ensues
          state_arg,
          next_state.as_mut_ptr(),
          next_logits.as_mut_ptr(),
        )
      };
    }

    Ok("hi")
  }
}
