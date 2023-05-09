use std::collections::{hash_map::Entry, HashMap};

use anyhow::{anyhow, Context, Result};
use frankenstein::{
  AsyncApi, AsyncTelegramApi, GetUpdatesParams, Message, SendMessageParams, UpdateContent,
};

struct BotOptions {
  model_path: String,
  tokens_path: String,
  api_token: String,
  n_threads: u32,
}

struct Bot<'a> {
  model: rwkv::Model,
  channels: HashMap<i64, rwkv::Session<'a>>,
  api: AsyncApi,
}

impl<'a> Bot<'a> {
  fn new(options: BotOptions) -> Result<Bot<'a>> {
    Ok(Bot {
      model: rwkv::Model::new(&options.model_path, &options.tokens_path, options.n_threads)?,
      channels: HashMap::new(),
      api: AsyncApi::new(&options.api_token),
    })
  }

  pub async fn run_bot_loop(&self) {
    let update_params_builder = GetUpdatesParams::builder();
    let mut update_params = update_params_builder.clone().build();

    loop {
      let result = self.api.get_updates(&update_params).await;

      log::debug!("result: {result:?}");

      match result {
        Ok(response) => {
          for update in response.result {
            if let UpdateContent::Message(message) = update.content {
              self.process_message(message).await;
            }
            update_params = update_params_builder
              .clone()
              .offset(update.update_id + 1)
              .build();
          }
        }
        Err(error) => {
          log::warn!("Failed to get updates: {error:?}");
        }
      }
    }
  }
  async fn process_message(&self, message: Message) {
    // message.from.unwrap().username.unwrap();
    // match message.entities.unwrap()[0].type_field {
    //   frankenstein::MessageEntityType::Mention
    // }
    let send_message_params = SendMessageParams::builder()
      .chat_id(message.chat.id)
      .text("hello")
      .reply_to_message_id(message.message_id)
      .build();

    if let Err(err) = self.api.send_message(&send_message_params).await {
      log::warn!("Failed to send message: {err:?}");
    }
  }

  fn get_or_create_session(&'a mut self, id: &i64) -> Result<&rwkv::Session<'a>> {
    match self.channels.entry(*id) {
      Entry::Occupied(occupied) => {
        // If the entry exists, return a reference to the existing ChatbotSession
        Ok(occupied.into_mut())
      }
      Entry::Vacant(vacant) => {
        // If the entry does not exist, create a new ChatbotSession and insert it
        let session = self.model.create_session()?;
        Ok(vacant.insert(session))
      }
    }
  }
}

#[tokio::main]
async fn main() -> Result<()> {
  env_logger::init();

  let model_path = std::env::var("MODEL_PATH")
    .with_context(|| "MODEL_PATH variable unset, set it to the path of the RWKV model")?;

  let tokens_path = std::env::var("TOKENS_PATH")
    .with_context(|| "TOKENS_PATH variable unset, set it to the path of the RWKV tokenizer json")?;

  let api_token =
    std::env::var("BOT_TOKEN").with_context(|| "BOT_TOKEN not set to telegram API token")?;

  let bot_options = BotOptions {
    model_path,
    tokens_path,
    api_token,
    n_threads: 4,
  };

  let bot = Bot::new(bot_options)
    .map_err(|e| anyhow!(e))
    .with_context(|| "Unable ot initialize BotState")?;

  Ok(bot.run_bot_loop().await)
}
