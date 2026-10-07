// Test-only experiment on the existing Head and Session. Never selected in a
// release engine; no extra vocabulary softmax or raw-argmax substitution.
use super::*;
use crate::{
    array::{clear_cache, synchronize},
    mtp::depth_policy::Policy,
};
use std::{
    cell::{Cell, RefCell},
    io::{Seek, SeekFrom, Write},
    time::Instant,
};

thread_local! {
    static POLICY: Cell<Option<Policy>> = const { Cell::new(None) };
    static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn policy() -> Option<Policy> {
    POLICY.get()
}

struct Scope;
impl Drop for Scope {
    fn drop(&mut self) {
        POLICY.set(None);
    }
}

fn with_policy<T>(policy: Policy, run: impl FnOnce() -> Result<T>) -> Result<T> {
    ensure!(POLICY.get().is_none(), "nested dynamic MTP scope");
    POLICY.set(Some(policy));
    SCORES.with_borrow_mut(Vec::clear);
    let _scope = Scope;
    run()
}

impl Head {
    pub(super) fn draft_scored(
        &mut self,
        target: &Qwen35Moe,
        hidden: &Array,
        anchor: u32,
        width: usize,
        policy: Policy,
    ) -> Result<(Vec<u32>, Cache)> {
        ensure!((1..=3).contains(&width), "invalid dynamic MTP width");
        ensure!(
            self.draft_head.is_none() && self.draft_ids.is_none(),
            "confidence screen only supports the legacy full-vocabulary head"
        );
        ensure!(
            (1..=16_777_216).contains(&self.layout.vocab_size),
            "IDs not exactly representable as f32"
        );
        let mut token = Array::from_u32(&[anchor], &[1, 1])?;
        let mut hidden = hidden.try_clone()?;
        let mut proposed = Vec::with_capacity(width);
        let mut exact = None;
        for index in 0..width {
            let mixed = self.mixed_embedding(&hidden, &target.mtp_embedding_ids(&token)?)?;
            hidden = self.residual(&mixed)?;
            if index == 0 {
                exact = Some(self.snapshot()?);
            }
            let logits =
                target.mtp_logits(&hidden.rms_norm(&self.norm, self.layout.rms_norm_eps)?)?;
            token = logits.log_probs()?.argmax()?.reshape(&[1, 1])?;
            let selected = logits
                .reshape(&[-1])?
                .take(&token.reshape(&[1])?, 0)?
                .astype(crate::array::Dtype::Float32)?;
            let id = token.reshape(&[1])?.astype(crate::array::Dtype::Float32)?;
            let pair = Array::concatenate(&[&id, &selected], 0)?.to_f32()?;
            ensure!(
                pair.len() == 2 && pair.iter().all(|value| value.is_finite()),
                "invalid scored draft readback"
            );
            ensure!(
                pair[0] >= 0. && pair[0] < self.layout.vocab_size as f32 && pair[0].fract() == 0.,
                "invalid scored token ID"
            );
            proposed.push(pair[0] as u32);
            SCORES.with_borrow_mut(|scores| scores.push(pair[1]));
            if !policy.continue_after(index + 1, width, pair[1]) {
                break;
            }
        }
        Ok((proposed, exact.context("missing dynamic MTP first cache")?))
    }
}

#[test]
fn scope_restores_after_errors_and_panics() -> Result<()> {
    let policy = Policy {
        thresholds: [8., 12.],
    };
    with_policy(policy, || {
        assert!(with_policy(policy, || Ok(())).is_err());
        assert!(
            std::thread::spawn(|| POLICY.get().is_none())
                .join()
                .unwrap()
        );
        Ok(())
    })?;
    assert!(POLICY.get().is_none());
    assert!(with_policy(policy, || -> Result<()> { anyhow::bail!("test error") }).is_err());
    assert!(POLICY.get().is_none());
    assert!(
        std::panic::catch_unwind(|| {
            let _ = with_policy(policy, || -> Result<()> { panic!("test panic") });
        })
        .is_err()
    );
    assert!(POLICY.get().is_none());
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn scope_drop_restores_actual_policy() {
    assert!(POLICY.get().is_none());
    POLICY.set(Some(Policy {
        thresholds: kani::any(),
    }));
    let scope = Scope;
    assert!(POLICY.get().is_some());
    drop(scope);
    assert!(POLICY.get().is_none());
    kani::cover!(POLICY.get().is_none());
}

fn checked_logits(value: &Array, width: usize) -> Result<Vec<u16>> {
    ensure!(
        value.shape() == [1, width as i32 + 1, 248320]
            && value.dtype() == crate::array::Dtype::Float16,
        "invalid dynamic target logits shape/dtype"
    );
    let bits = value.to_f16_bits()?;
    ensure!(
        bits.len() == (width + 1) * 248320
            && bits
                .iter()
                .all(|&value| half::f16::from_bits(value).is_finite()),
        "invalid dynamic logits"
    );
    Ok(bits)
}

fn draft_bytes(head: &Head) -> Result<(Vec<u8>, Vec<u8>)> {
    let state = head.snapshot()?;
    for value in [&state.keys, &state.values] {
        ensure!(
            value.dtype() == crate::array::Dtype::Float16 && value.byte_len()? > 0,
            "invalid dynamic cache"
        );
        ensure!(
            value
                .to_f16_bits()?
                .iter()
                .all(|&value| half::f16::from_bits(value).is_finite()),
            "nonfinite dynamic cache"
        );
    }
    Ok((state.keys.to_bytes()?, state.values.to_bytes()?))
}

#[test]
#[ignore = "requires physical Apple GPU and local dense Qwen/MTP; serial experiment"]
fn real_adaptive_depth_screen() -> Result<()> {
    use crate::qwen35::profile;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var("MLXL3_DYNAMIC_REPORT")?)?;
    let mut cells = Vec::new();
    let validation_only = std::env::var("MLXL3_DYNAMIC_VALIDATION_ONLY").as_deref() == Ok("1");
    let persist = |file: &mut File,
                   cells: &[serde_json::Value],
                   status: &str,
                   error: Option<String>|
     -> Result<()> {
        let bytes = serde_json::to_vec_pretty(
            &serde_json::json!({"status":status,"parity":status=="complete","cells":cells,"error":error,
            "boundary":"snapshot-restored real MTP rounds; confidence overhead and offline sweep, not sustained bridge generation"}),
        )?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&bytes)?;
        file.set_len(bytes.len() as u64)?;
        file.sync_all()?;
        Ok(())
    };
    persist(&mut file, &cells, "running", None)?;
    let result = (|| -> Result<()> {
        let path = std::env::var("MLXL3_MTP_TEST_MODEL")?;
        let tokenizer = crate::tokenizer::ChatTokenizer::load(Path::new(&path))?;
        let mut target = Qwen35Moe::load(Path::new(&path))?;
        ensure!(
            target.mtp_layout().hidden_size == 5120 && target.mtp_layout().vocab_size == 248320,
            "dynamic screen requires Qwen27B"
        );
        let mut head = Head::load(Path::new(&std::env::var("MLXL3_MTP_TEST_HEAD")?), &target)?;
        let context = 4096;
        for (workload, text) in [
            (
                "code",
                "Implement a bounded queue in Rust using Result. Show the push method and explain overflow handling.\n",
            ),
            (
                "prose",
                "Explain how the seasons influence a forest ecosystem. Include concrete examples and clear transitions.\n",
            ),
            (
                "fr",
                "Explique en français comment comparer deux moteurs d'inférence avec des mesures reproductibles.\n",
            ),
            (
                "structured",
                "Return valid JSON with three objects containing name, integer score and boolean enabled.\n",
            ),
        ] {
            target.reset();
            head.reset()?;
            let seed = tokenizer.encode(text)?;
            ensure!(!seed.is_empty(), "empty dynamic workload");
            let prefix = seed
                .iter()
                .copied()
                .cycle()
                .take(context)
                .collect::<Vec<_>>();
            let mut anchor = 0;
            for start in (0..context).step_by(128) {
                let end = (start + 128).min(context);
                let previous = if start > 0 {
                    Some(target.mtp_hidden()?.try_clone()?)
                } else {
                    None
                };
                let (logits, raw) = target.forward_mtp(&prefix[start..end])?;
                let n = (end - start) as i32;
                if let Some(previous) = previous {
                    head.extend_cache(
                        &target,
                        &Array::concatenate(&[&previous, &raw.slice(1, 0, n - 1)?], 1)?,
                        &prefix[start..end],
                    )?;
                } else {
                    head.extend_cache(&target, &raw.slice(1, 0, n - 1)?, &prefix[1..end])?;
                }
                anchor = logits.chat_greedy_ids()?[0];
                clear_cache()?;
            }
            let base = target.snapshot()?;
            let mut draft_base = head.snapshot()?;
            let policies = [
                Policy {
                    thresholds: [f32::INFINITY; 2],
                },
                Policy {
                    thresholds: [f32::NEG_INFINITY, f32::INFINITY],
                },
                Policy {
                    thresholds: [f32::NEG_INFINITY; 2],
                },
                Policy {
                    thresholds: [8., 12.],
                },
                Policy {
                    thresholds: [16., 20.],
                },
                Policy {
                    thresholds: [f32::NEG_INFINITY; 2],
                },
                Policy {
                    thresholds: [f32::NEG_INFINITY; 2],
                },
                Policy {
                    thresholds: [f32::NEG_INFINITY; 2],
                },
            ];
            let mut quality = Vec::new();
            for (index, policy) in policies.into_iter().enumerate() {
                target.restore(base.clone())?;
                head.restore(draft_base)?;
                draft_base = head.snapshot()?;
                clear_cache()?;
                let mut actual = Session::new(3)?;
                let scope = profile::Scope::capture()?;
                let output_budget = match index {
                    5 => 2,
                    6 => 3,
                    _ => 256,
                };
                let context_limit = if index == 7 {
                    context + 2
                } else {
                    context + 256
                };
                let first = with_policy(policy, || {
                    actual.advance(
                        &mut target,
                        &mut head,
                        anchor,
                        context_limit,
                        output_budget,
                        |_| false,
                    )
                })?;
                let width = actual.proposed;
                let logits = checked_logits(&scope.logits()?, width)?;
                drop(scope);
                let state = profile::StateReference::save(&target.snapshot()?)?;
                let draft = draft_bytes(&head)?;
                let scores = SCORES.with_borrow(Clone::clone);
                ensure!(
                    scores.len() == width && (1..=3).contains(&width),
                    "invalid adaptive trace width"
                );
                if index < 3 {
                    ensure!(width == index + 1, "forced adaptive width differs");
                }
                if index >= 5 {
                    ensure!(
                        width == if index == 6 { 2 } else { 1 },
                        "adaptive budget cap differs"
                    );
                }
                target.restore(base.clone())?;
                head.restore(draft_base)?;
                draft_base = head.snapshot()?;
                clear_cache()?;
                let mut stock = Session::new(width)?;
                let scope = profile::Scope::capture()?;
                let expected = stock.advance(
                    &mut target,
                    &mut head,
                    anchor,
                    context_limit,
                    output_budget,
                    |_| false,
                )?;
                ensure!(
                    first == expected
                        && actual.pending == stock.pending
                        && (actual.proposed, actual.accepted, actual.blocks)
                            == (stock.proposed, stock.accepted, stock.blocks),
                    "adaptive IDs/pending/counters differ"
                );
                ensure!(
                    checked_logits(&scope.logits()?, width)? == logits,
                    "adaptive full logits differ"
                );
                drop(scope);
                state.compare(&target.snapshot()?)?;
                ensure!(
                    draft_bytes(&head)? == draft,
                    "adaptive repaired draft cache differs"
                );
                quality.push(serde_json::json!({"policy_index":index,"actual_depth":width,"output_budget":output_budget,"context_limit":context_limit,"scores":scores,"accepted":actual.accepted,"bit_exact":true,"state_arrays":129}));
            }
            if validation_only {
                cells.push(serde_json::json!({"workload":workload,"context":context,"quality":quality,"validation_only":true,"timings":[],"offline_trace":[]}));
                persist(&mut file, &cells, "running", None)?;
                target.restore(base)?;
                head.restore(draft_base)?;
                clear_cache()?;
                println!("D {workload}:32-case campaign quality only, exact legacy head");
                continue;
            }
            let mut timings = Vec::new();
            // Fixed depths provide the relevant controls. Always-D3 scored
            // isolates the new readback cost without confounding early stops.
            for depth in 1..=3 {
                let mut samples = [Vec::new(), Vec::new()];
                let mut trace = None;
                for repeat in 0..11 {
                    for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                        target.restore(base.clone())?;
                        head.restore(draft_base)?;
                        draft_base = head.snapshot()?;
                        clear_cache()?;
                        synchronize()?;
                        let mut session = Session::new(depth)?;
                        let start = Instant::now();
                        if arm == 0 {
                            session.advance(
                                &mut target,
                                &mut head,
                                anchor,
                                context + 256,
                                256,
                                |_| false,
                            )?;
                        } else {
                            with_policy(
                                Policy {
                                    thresholds: [f32::NEG_INFINITY; 2],
                                },
                                || {
                                    session.advance(
                                        &mut target,
                                        &mut head,
                                        anchor,
                                        context + 256,
                                        256,
                                        |_| false,
                                    )
                                },
                            )?;
                        }
                        synchronize()?;
                        let seconds = start.elapsed().as_secs_f64();
                        if repeat >= 3 {
                            samples[arm].push(seconds);
                        }
                        if arm == 1 {
                            trace = Some(
                                serde_json::json!({"scores":SCORES.with_borrow(Clone::clone),"accepted":session.accepted,"delivered":session.pending.len()+1,"target_rows":depth+1}),
                            );
                        }
                    }
                }
                timings.push(
                    serde_json::json!({"depth":depth,"samples_seconds":samples,"trace":trace}),
                );
            }
            target.restore(base)?;
            head.restore(draft_base)?;
            clear_cache()?;
            let mut trace = Vec::new();
            let mut session = Session::new(3)?;
            for _ in 0..8 {
                let before = (session.proposed, session.accepted);
                let start = Instant::now();
                let first = with_policy(
                    Policy {
                        thresholds: [f32::NEG_INFINITY; 2],
                    },
                    || {
                        session.advance(&mut target, &mut head, anchor, context + 256, 256, |_| {
                            false
                        })
                    },
                )?;
                let mut delivered = vec![first];
                delivered.extend(session.pending.drain(..));
                synchronize()?;
                trace.push(serde_json::json!({"scores":SCORES.with_borrow(Clone::clone),"accepted":session.accepted-before.1,"proposed":session.proposed-before.0,"delivered":delivered.len(),"seconds":start.elapsed().as_secs_f64()}));
                anchor = *delivered.last().unwrap();
                clear_cache()?;
            }
            cells.push(serde_json::json!({"workload":workload,"context":context,"quality":quality,"timings":timings,"offline_trace":trace}));
            persist(&mut file, &cells, "running", None)?;
            println!(
                "D {workload}: full logits/129 states/draft/IDs exact, fixed/scored pairs complete"
            );
        }
        ensure!(cells.len() == 4, "incomplete dynamic workloads");
        Ok(())
    })();
    persist(
        &mut file,
        &cells,
        if result.is_ok() { "complete" } else { "failed" },
        result.as_ref().err().map(ToString::to_string),
    )?;
    result
}
