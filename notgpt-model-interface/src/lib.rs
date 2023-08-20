use thiserror::Error;
// Type token = u32

#[derive(Error, Debug)]
pub enum NotGptError {
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

pub trait GenericModel {
  const STOP_TOKEN: u32;

  type SessionState: Default;

  fn predict_logits(
    &mut self,
    session: &mut Self::SessionState,
    input_tokens: &Vec<u32>,
  ) -> Result<Vec<f32>, NotGptError>;

  fn load_session_state(&mut self, state: &mut Self::SessionState);

  fn save_session_state(&self, state: &mut Self::SessionState);
}
