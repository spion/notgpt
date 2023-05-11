use std::cmp::Ordering;

use rand::{distributions::WeightedIndex, prelude::Distribution};

fn softmax(vec: &Vec<f32>) -> Vec<f32> {
  let denominator: f32 = vec.iter().map(|val| val.exp()).sum();

  vec.iter().map(|val| val.exp() / denominator).collect()
}

fn random_choice(items: &Vec<(usize, f32)>) -> usize {
  let mut rng = rand::thread_rng();

  let dist: WeightedIndex<f32> =
    WeightedIndex::new(items.iter().map(|(_id, p)| p)).expect("Invalid probabilities");

  let (id, _p) = items
    .get(dist.sample(&mut rng))
    .expect("Sample must exist in original Vec");

  *id
}

fn sample_top_p(probabilities_i: &Vec<(usize, f32)>, top_p: f32) -> Vec<(usize, f32)> {
  let mut probabilities = probabilities_i.clone();

  probabilities.sort_by(|(_id1, p1), (_id2, p2)| p2.partial_cmp(p1).unwrap_or(Ordering::Equal));

  let mut running_sum: f32 = 0.0;
  probabilities = probabilities
    .into_iter()
    .take_while(|(_id, p)| {
      let take_more = running_sum < top_p;
      running_sum += p;
      take_more
    })
    .collect();

  probabilities
}

fn adjust_temp(probabilities: &mut Vec<(usize, f32)>, temp: f32) {
  for (_, p) in probabilities {
    *p = p.powf(1.0 / temp);
  }
}

fn apply_repetition_penalty(
  probabilities: &mut Vec<(usize, f32)>,
  previous_tokens: &Vec<u32>,
  repeat_penalty: f32,
  repeat_len: usize,
) {
  if previous_tokens.len() <= 0 {
    return;
  }
  for (id, probability) in probabilities {
    for k in (previous_tokens.len() - repeat_len).max(0)..previous_tokens.len() {
      if previous_tokens[k] == *id as u32 {
        *probability = *probability / repeat_penalty;
      }
    }
  }
}

pub struct SampleOptions {
  temp: f32,
  top_p: f32,
  repeat_penalty: f32,
  repeat_len: usize,
}

impl Default for SampleOptions {
  fn default() -> Self {
    Self {
      temp: 0.7,
      top_p: 0.6,
      repeat_penalty: 1.176,
      repeat_len: 64,
    }
  }
}

pub fn sample_token(logits: &Vec<f32>, previous_tokens: &Vec<u32>, opts: SampleOptions) -> usize {
  if opts.temp <= f32::EPSILON {
    let (id, _val) = logits
      .iter()
      .enumerate()
      .filter(|(_id, p)| !f32::is_nan(**p) && f32::is_finite(**p))
      .max_by(|(_id1, p1), (_id2, p2)| p1.partial_cmp(p2).unwrap_or(Ordering::Equal))
      .expect("Max must be there");

    return id;
  }

  log::trace!("Probabilities size before filtering {}", logits.len());

  let mut probabilities = softmax(&logits)
    .into_iter()
    .enumerate()
    .filter(|(_id, p)| !f32::is_nan(*p) && f32::is_finite(*p))
    .collect::<Vec<_>>();

  log::trace!("Probabilities size after filtering {}", probabilities.len());

  if opts.temp <= 1.0 - f32::EPSILON {
    log::debug!("Applying temperature");
    adjust_temp(&mut probabilities, opts.temp);
  }

  if opts.repeat_penalty > 1.0 + f32::EPSILON {
    log::debug!("Applying repetition penalty");
    apply_repetition_penalty(
      &mut probabilities,
      previous_tokens,
      opts.repeat_penalty,
      opts.repeat_len,
    );
  }

  if opts.top_p < 1.0 - f32::EPSILON {
    probabilities = sample_top_p(&probabilities, opts.top_p);
  }

  log::trace!(
    "Probabilities size before random_choice {}",
    probabilities.len()
  );
  random_choice(&probabilities)
}

#[cfg(test)]
mod tests {
  use crate::token_functions::apply_repetition_penalty;

  #[test]
  fn it_works() {
    let mut probs = vec![0.2, 0.3, 0.5, 0.1]
      .into_iter()
      .enumerate()
      .collect::<Vec<(usize, f32)>>();

    let used = vec![2, 3, 0, 1, 0];

    apply_repetition_penalty(&mut probs, &used, 2.0, 3);

    assert_eq!(probs, [(0, 0.1), (1, 0.3), (2, 0.5), (3, 0.1)]);
  }
}
