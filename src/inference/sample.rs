//! CPU-side sampling helpers. Just argmax for the MVP.

pub fn argmax(logits: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_highest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax(&[-3.0, -2.0, -1.0]), 2);
    }

    #[test]
    fn argmax_first_on_tie() {
        assert_eq!(argmax(&[1.0, 1.0, 1.0]), 0);
    }
}
