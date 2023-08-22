FROM alpine:3.18
ADD ./target/release/notgpt-telegram /notgpt-telegram
ENTRYPOINT /notgpt-telegram