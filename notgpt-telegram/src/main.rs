use std::{
  collections::{hash_map::Entry, HashMap},
  sync::Arc,
};

use anyhow::{anyhow, Context, Result};
use frankenstein::{
  AsyncApi, AsyncTelegramApi, GetUpdatesParams, Message, SendMessageParams, UpdateContent,
};
use tokio::sync::Mutex;

struct BotOptions {
  pub model_path: String,
  pub tokens_path: String,
  pub api_token: String,
  pub n_threads: u32,
  pub username: String,
}

struct Bot {
  model: rwkv::Model,
  channels: HashMap<i64, rwkv::Session>,
  api: AsyncApi,
  username: String,
}

impl Bot {
  fn new(options: BotOptions) -> Result<Bot> {
    Ok(Bot {
      model: rwkv::Model::new(&options.model_path, &options.tokens_path, options.n_threads)?,
      channels: HashMap::new(),
      api: AsyncApi::new(&options.api_token),
      username: options.username,
    })
  }

  pub fn process_messages(
    &mut self,
    username: &String,
    messages: Vec<Message>,
  ) -> Result<Vec<SendMessageParams>> {
    let mut results: Vec<SendMessageParams> = vec![];

    for message in messages {
      let response = self.process_message(username, message)?;
      match response {
        Some(res) => {
          results.push(res);
        }
        None => {}
      }
    }

    Ok(results)
  }

  pub async fn run_bot_loop(bot_ref: Arc<Mutex<Bot>>) {
    let update_params_builder = GetUpdatesParams::builder();
    let mut update_params = update_params_builder.clone().build();
    let api = { bot_ref.lock().await.api.clone() };

    loop {
      let result = api.get_updates(&update_params).await;

      match result {
        Ok(response) => {
          // let response_len = response.result.len();
          log::debug!("Response is {:?}", response);

          let messages = response
            .result
            .iter()
            .flat_map(|update| match update.content.clone() {
              UpdateContent::Message(message) if message.text.is_some() => vec![message],
              _ => {
                log::debug!("Non message event {:?}", update);
                vec![]
              }
            })
            .collect::<Vec<_>>();

          let username = { bot_ref.lock().await.username.clone() };

          let responses = {
            let mut bot = bot_ref.lock().await;
            bot.process_messages(&username, messages)
          };
          match responses {
            Err(err) => {
              log::warn!("Failed to process message:s {err:?}");
            }
            Ok(messages) => {
              for message in messages {
                log::debug!("Sending message: {:?}", message);
                match api.send_message(&message).await {
                  Err(err) => {
                    log::warn!("Failed to send message: {err:?}");
                  }
                  _ => {}
                }
              }
            }
          }

          let last_update = response.result.iter().last();

          update_params = match last_update {
            Some(update) => update_params_builder
              .clone()
              .offset(update.update_id + 1)
              .build(),
            None => update_params,
          }
        }

        Err(error) => {
          log::warn!("Failed to get updates: {error:?}");
        }
      }
    }
  }

  fn process_message(
    &mut self,
    username: &String,
    message: Message,
  ) -> Result<Option<SendMessageParams>> {
    let session = self.get_or_create_session(&message.chat.id)?;

    let mut was_mentioned = false;

    for item in message.entities.clone().unwrap_or_default() {
      match item.type_field {
        frankenstein::MessageEntityType::Mention => {
          let mention_content = message.text.clone().unwrap()
            [(item.offset as usize)..((item.offset + item.length) as usize)]
            .to_string();

          log::debug!("Found mention: {}", mention_content);

          if &mention_content == username || mention_content == format!("@{}", username) {
            was_mentioned = true;
          }
        }
        _ => {}
      }
    }

    log::debug!("Processing {:?}", &message);

    let user = message.from.unwrap();
    let text = message.text.unwrap();

    let text = format!(
      "{0}: {1}\n\n",
      user.username.unwrap_or(user.first_name),
      text
    );
    session.consume_text(&text)?;

    if was_mentioned {
      let additional_text = format!("{}: ", username.to_string());
      session.consume_text(&additional_text)?;

      let output = session.produce_text()?;

      let send_message_params = SendMessageParams::builder()
        .chat_id(message.chat.id)
        .text(output)
        .reply_to_message_id(message.message_id)
        .build();
      Ok(Some(send_message_params))
    } else {
      Ok(None)
    }
  }

  fn initial_prompt(bot_username: &String) -> String {
    format!(
      r#"

The following is a verbose and detailed Telegram channel conversation between multiple people and an AI assistant called {0}. The people in the channel may mention {0} by including his name with an @ prefix, like so: @{0}. {0} is intelligent, knowledgeable, wise and polite, and when mentioned responds appropriately taking into account both the message and the previous context.

Jack Jones: What year was the french revolution?

Jill Someone: Not sure. Could we ask @{0}?

{0}: The French Revolution started in 1789, and lasted 10 years until 1799.

The Math Nerd: Wow, thats pretty impressive.

Moe: What is?

The Math Nerd: Look. @{0} 3+5=?

{0}: The answer is 8.

Moe: can you gues who I'll marry @{0}?

{0}: Only if you tell me more about yourself - what are your interests?

The Math Nerd: hah, he got you.

The Math Nerd: solve for a: 9-a=2 @{0}

{0}: The answer is a = 7, because 9 - 7 = 2.

User: wat is lhc?

Jill Someone: If you are talking to {0} you have to mention it. @{0} can you answer?

{0}: LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

"#,
      bot_username
    )
  }

  fn get_or_create_session(&mut self, id: &i64) -> Result<&mut rwkv::Session> {
    match self.channels.entry(*id) {
      Entry::Occupied(occupied) => {
        // If the entry exists, return a reference to the existing ChatbotSession
        Ok(occupied.into_mut())
      }
      Entry::Vacant(vacant) => {
        let initial_prompt = Bot::initial_prompt(&self.username);
        // If the entry does not exist, create a new ChatbotSession and insert it
        let session = self.model.create_session_custom(&rwkv::SessionOptions {
          prompt: rwkv::Prompt {
            prompt: initial_prompt,
            ..Default::default()
          },
          ..Default::default()
        })?;
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

  let api_token = std::env::var("BOT_TOKEN")
    .with_context(|| "BOT_TOKEN not set to telegram API token, please configure")?;

  let username = std::env::var("BOT_USERNAME")
    .with_context(|| "BOT_USERNAME not set to telegram bot username")?;

  let bot_options = BotOptions {
    model_path,
    tokens_path,
    api_token,
    n_threads: 4,
    username,
  };

  let bot = Bot::new(bot_options)
    .map_err(|e| anyhow!(e))
    .with_context(|| "Unable ot initialize BotState")?;

  let bot_ref = Arc::new(Mutex::new(bot));
  Bot::run_bot_loop(bot_ref).await;

  Ok(())
}
