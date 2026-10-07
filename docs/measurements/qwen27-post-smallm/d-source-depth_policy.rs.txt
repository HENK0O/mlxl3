//! Test-only candidate; the production draft chain stays lazy and unchanged.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Policy {
    pub thresholds: [f32; 2],
}

impl Policy {
    pub(crate) fn continue_after(self, completed: usize, cap: usize, score: f32) -> bool {
        (1..=3).contains(&cap)
            && (1..=2).contains(&completed)
            && completed < cap
            && score.is_finite()
            && score >= self.thresholds[completed - 1]
    }
}

#[cfg(kani)]
#[kani::proof]
fn adaptive_depth_never_crosses_cap_or_nonfinite_score() {
    let policy = Policy {
        thresholds: kani::any(),
    };
    let completed: usize = kani::any();
    let cap: usize = kani::any();
    let score: f32 = kani::any();
    let advance = policy.continue_after(completed, cap, score);
    if advance {
        assert!((1..=2).contains(&completed));
        assert!((2..=3).contains(&cap) && completed < cap);
        assert!(score.is_finite() && score >= policy.thresholds[completed - 1]);
        assert!(completed + 1 <= cap);
    }
    if completed == 1 && cap >= 2 && cap <= 3 && score.is_finite() {
        assert_eq!(advance, score >= policy.thresholds[0]);
    }
    kani::cover!(advance && completed == 1);
    kani::cover!(advance && completed == 2);
    kani::cover!(!advance && !score.is_finite());
    kani::cover!(!advance && completed == cap);
}

#[test]
fn policy_checks_budgets_thresholds_and_nonfinite_values() {
    let policy = Policy {
        thresholds: [8., 12.],
    };
    for cap in [0, 1, 2, 3, 4, usize::MAX] {
        for completed in [0, 1, 2, 3, 4, usize::MAX] {
            for score in [f32::NEG_INFINITY, f32::NAN, 7., 8., 11., 12., f32::INFINITY] {
                let expected = match (completed, cap) {
                    (1, 2 | 3) => score.is_finite() && score >= 8.,
                    (2, 3) => score.is_finite() && score >= 12.,
                    _ => false,
                };
                assert_eq!(policy.continue_after(completed, cap, score), expected);
            }
        }
    }
    assert!(
        !Policy {
            thresholds: [f32::NAN, 0.]
        }
        .continue_after(1, 3, 12.)
    );
    assert!(
        !Policy {
            thresholds: [f32::INFINITY, 0.]
        }
        .continue_after(1, 3, 12.)
    );
    assert!(
        Policy {
            thresholds: [f32::NEG_INFINITY; 2]
        }
        .continue_after(2, 3, 12.)
    );
}
