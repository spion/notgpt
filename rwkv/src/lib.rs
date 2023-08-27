// See https://github.com/saharNooby/rwkv.cpp/blob/master/rwkv/rwkv_cpp_model.py

use std::{
  ffi::{c_char, CString},
  path::Path,
  ptr,
};

use notgpt_model_interface::{GenericModel, NotGptError};
use rwkv_sys;

pub struct Model {
  ctx: *mut rwkv_sys::rwkv_context,
  state_buffer_element_count: usize,
  logits_buffer_element_count: usize,
}

impl Drop for Model {
  fn drop(&mut self) {
    unsafe { rwkv_sys::rwkv_free(self.ctx) }
  }
}

// static END_OF_LINE_TOKEN: u32 = 187;

pub struct State {
  current_state: Option<Vec<f32>>,
}

impl Default for State {
  fn default() -> Self {
    Self {
      current_state: Default::default(),
    }
  }
}

impl GenericModel for Model {
  const STOP_TOKEN: u32 = 0;

  type SessionState = State;

  fn predict_logits(
    &mut self,
    session: &mut Self::SessionState,
    input_tokens: &[u32],
  ) -> Result<Vec<f32>, NotGptError> {
    let mut end_result = Err(NotGptError::TokenPredictionError);

    for token in input_tokens {
      let (state, logits) = self.predict(&session.current_state, *token)?;
      session.current_state = Some(state);
      end_result = Ok(logits)
    }

    return end_result;
  }

  fn load_session_state(&mut self, _state: &mut Self::SessionState) {}

  fn save_session_state(&self, _state: &mut Self::SessionState) {}
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

    let ctx = unsafe { rwkv_sys::rwkv_init_from_file(raw_ptr, n_threads) };

    let state_size = unsafe { rwkv_sys::rwkv_get_state_buffer_element_count(ctx) } as usize;
    let logits_size = unsafe { rwkv_sys::rwkv_get_logits_buffer_element_count(ctx) } as usize;

    Ok(Model {
      ctx: ctx,

      state_buffer_element_count: state_size,
      logits_buffer_element_count: logits_size,
    })
  }

  pub fn predict(
    &self,
    current_state: &Option<Vec<f32>>,
    token: u32,
  ) -> Result<(Vec<f32>, Vec<f32>), NotGptError> {
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
      Ok((next_state, next_logits))
    } else {
      Err(NotGptError::TokenPredictionError)
    }
  }
}
