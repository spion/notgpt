// See https://github.com/saharNooby/rwkv.cpp/blob/master/rwkv/rwkv_cpp_shared_library.py

use std::{
  ffi::{c_char, CString},
  path::Path,
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
  tokenizer: Tokenizer,
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

    Ok(RWKVModel {
      ctx: ctx,
      tokenizer: tokenizer,
    })
  }

  pub fn predict<'s, E>(self, text: &str) -> Result<&str, RWKVError> {
    let encoding = self.tokenizer.encode(text, true).map_err(|e| RWKVError::TokenEncodeError { source:  e })?;

    for token in encoding.get_ids() {
      // rwkv_sys::rwkv_eval(self.ctx,token, ...)
    }



    Ok("hi")
  }
}
