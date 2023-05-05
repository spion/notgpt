use anyhow::Result;
use frankenstein::{
  AsyncApi, AsyncTelegramApi, GetUpdatesParams, Message, SendMessageParams, UpdateContent,
};

// use frankenstein::api::
static TOKEN: &str = "API_TOKEN";

#[tokio::main]
async fn main() -> Result<()> {
  let api = AsyncApi::new(TOKEN);

  let update_params_builder = GetUpdatesParams::builder();
  let mut update_params = update_params_builder.clone().build();

  loop {
    let result = api.get_updates(&update_params).await;

    println!("result: {result:?}");

    match result {
      Ok(response) => {
        for update in response.result {
          if let UpdateContent::Message(message) = update.content {
            let api_clone = api.clone();

            tokio::spawn(async move {
              process_message(message, api_clone).await;
            });
          }
          update_params = update_params_builder
            .clone()
            .offset(update.update_id + 1)
            .build();
        }
      }
      Err(error) => {
        println!("Failed to get updates: {error:?}");
      }
    }
  }
}

async fn process_message(message: Message, api: AsyncApi) {
  // message.from.unwrap().username.unwrap();
  // match message.entities.unwrap()[0].type_field {
  //   frankenstein::MessageEntityType::Mention
  // }
  let send_message_params = SendMessageParams::builder()
    .chat_id(message.chat.id)
    .text("hello")
    .reply_to_message_id(message.message_id)
    .build();

  if let Err(err) = api.send_message(&send_message_params).await {
    println!("Failed to send message: {err:?}");
  }
}
