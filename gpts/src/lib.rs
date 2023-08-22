// See https://github.com/saharNooby/rwkv.cpp/blob/master/rwkv/rwkv_cpp_model.py

// See also https://github.com/ggerganov/llama.cpp/pull/2304 for llama 2
use core::slice;
use std::{
  ffi::{c_char, CString},
  path::Path,
};

use gpts_sys::{self};
use notgpt_model_interface::{GenericModel, NotGptError};
use tokenizers::Tokenizer;

pub struct Model {
  ctx: *mut gpts_sys::llama_context,
  // state_buffer_element_count: usize,
  logits_buffer_element_count: usize,
  n_threads: u32,
}

impl Drop for Model {
  fn drop(&mut self) {
    unsafe { gpts_sys::llama_free(self.ctx) }
  }
}

pub struct State {
  n_past: u32,
  current_state: Vec<u8>,
}

impl Default for State {
  fn default() -> Self {
    Self {
      n_past: 0,
      current_state: vec![],
    }
  }
}

impl GenericModel for Model {
  const STOP_TOKEN: u32 = 2;

  type SessionState = State;

  fn predict_logits(
    &mut self,
    session: &mut Self::SessionState,
    input_tokens: &Vec<u32>,
  ) -> Result<Vec<f32>, NotGptError> {
    let success = unsafe {
      gpts_sys::llama_eval(
        self.ctx,
        input_tokens.as_ptr() as *const i32,
        1,
        session.n_past as i32,
        self.n_threads as i32,
      )
    };

    if success == 0 {
      let logits = unsafe { gpts_sys::llama_get_logits(self.ctx) };
      let logits_vec =
        unsafe { slice::from_raw_parts(logits, self.logits_buffer_element_count).to_vec() };

      session.n_past = session.n_past + 1;
      Ok(logits_vec)
    } else {
      Err(NotGptError::TokenPredictionError)
    }
  }

  fn load_session_state(&mut self, s: &mut Self::SessionState) {
    let state = &mut s.current_state;
    let ssize = unsafe { gpts_sys::llama_get_state_size(self.ctx) };
    if state.len() > 0 {
      unsafe {
        log::debug!("Loading state {}", ssize);
        gpts_sys::llama_set_state_data(self.ctx, state.as_mut_ptr());
        log::debug!("State loaded");
      }
    }
  }

  fn save_session_state(&self, s: &mut Self::SessionState) {
    let state = &mut s.current_state;

    let ssize = unsafe { gpts_sys::llama_get_state_size(self.ctx) };
    if state.len() != ssize {
      log::debug!("Resizing state to {}", ssize);
      state.resize(ssize, 0);
    }
    log::debug!("Saving state 2");
    unsafe {
      gpts_sys::llama_copy_state_data(self.ctx, state.as_mut_ptr());
      log::debug!("State saved");
    }
  }
}

impl Model {
  pub fn new<P: AsRef<Path>>(model_path: &P, n_threads: u32) -> Result<Model, NotGptError> {
    // let test = model_path.into<String>();
    let model_str = model_path
      .as_ref()
      .to_str()
      .expect("Failed to convert Path to str");

    let cstring = CString::new(model_str).expect("Failed to create CString");
    let raw_ptr: *const c_char = cstring.as_ptr();

    let params = gpts_sys::llama_context_params {
      logits_all: false,
      n_ctx: 4096,
      // n_batch: 8,
      ..unsafe { gpts_sys::llama_context_default_params() }
    };

    let ctx = unsafe { gpts_sys::llama_init_from_file(raw_ptr, params) };

    // let state_size = unsafe { rwkv_sys::rwkv_get_state_buffer_element_count(ctx) } as usize;
    let logits_size = unsafe { gpts_sys::llama_n_vocab(ctx) } as usize;

    // gpts_sys::llama_tokenize(
    Ok(Model {
      ctx: ctx,
      n_threads,
      logits_buffer_element_count: logits_size,
    })
  }
}
