//! A1: single K/V append with unchanged per-row SDPA. Experimental and OFF
//! unless explicitly enabled; only the measured dense M5 geometry is eligible.
use super::*;
#[cfg(test)]
use std::cell::Cell;

#[derive(Clone, Copy, Debug)]
pub(super) enum Preparation {
    BatchAll,
    #[cfg(test)]
    ScalarRope,
    #[cfg(test)]
    ScalarNormRope,
}

impl Preparation {
    fn scalar_norm(self) -> bool {
        #[cfg(test)]
        {
            matches!(self, Self::ScalarNormRope)
        }
        #[cfg(not(test))]
        {
            false
        }
    }
}

pub(super) fn selection(
    attention: &Attention,
    x: &Array,
    time: i32,
) -> Result<Option<Preparation>> {
    #[cfg(test)]
    if let Some(mode) = PREPARATION.get() {
        return Ok(mode);
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED
        .get_or_init(|| std::env::var("MLXL3_EXPERIMENTAL_BATCHED_VERIFY").as_deref() == Ok("1"));
    if !enabled {
        return Ok(None);
    }
    let past = attention.keys.as_ref().map_or(0, |keys| keys.shape()[2]);
    // Reject shape misses before querying the Metal device.
    if !crate::contracts::use_batched_verify_attention(
        time,
        past,
        true,
        x.shape()[2],
        attention.heads,
        attention.kv_heads,
        attention.head_dim,
    ) {
        return Ok(None);
    }
    Ok(crate::array::is_m5_gpu()?.then_some(Preparation::BatchAll))
}

#[cfg(test)]
thread_local! {
    static PREPARATION: Cell<Option<Option<Preparation>>> = const { Cell::new(None) };
}

#[cfg(test)]
pub(super) fn current() -> Option<Preparation> {
    PREPARATION.get().flatten()
}

#[cfg(test)]
pub(super) fn with_preparation<T>(mode: Option<Preparation>, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<Option<Preparation>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PREPARATION.set(self.0);
        }
    }
    let _restore = Restore(PREPARATION.replace(Some(mode)));
    f()
}

impl Attention {
    pub(super) fn forward_prepared(&mut self, x: &Array, mode: Preparation) -> Result<Array> {
        let time = x.shape()[1]; // Validated by the production caller.
        let base = self.keys.as_ref().map_or(0, |keys| keys.shape()[2]);
        ensure!(
            base >= 0 && base.checked_add(time).is_some(),
            "KV offset overflow"
        );
        let qkv = self.qkv.forward(x)?;
        let q_gate = qkv[0].reshape(&[1, time, self.heads, self.head_dim * 2])?;
        let q_raw = q_gate.slice(3, 0, self.head_dim)?;
        let k_raw = qkv[1].reshape(&[1, time, self.kv_heads, self.head_dim])?;
        let gate = q_gate
            .slice(3, self.head_dim, self.head_dim * 2)?
            .reshape(&[1, time, self.heads * self.head_dim])?;
        let (q, k) = if mode.scalar_norm() {
            let mut qs = Vec::new();
            let mut ks = Vec::new();
            for row in 0..time {
                qs.push(
                    q_raw
                        .slice(1, row, row + 1)?
                        .rms_norm(&self.q_norm, self.eps)?,
                );
                ks.push(
                    k_raw
                        .slice(1, row, row + 1)?
                        .rms_norm(&self.k_norm, self.eps)?,
                );
            }
            (
                Array::concatenate(&qs.iter().collect::<Vec<_>>(), 1)?,
                Array::concatenate(&ks.iter().collect::<Vec<_>>(), 1)?,
            )
        } else {
            (
                q_raw.rms_norm(&self.q_norm, self.eps)?,
                k_raw.rms_norm(&self.k_norm, self.eps)?,
            )
        };
        let q = q.transpose(&[0, 2, 1, 3])?;
        let k = k.transpose(&[0, 2, 1, 3])?;
        let (q, k) = if matches!(mode, Preparation::BatchAll) {
            (
                q.rope(self.rope_dims, self.theta, base)?,
                k.rope(self.rope_dims, self.theta, base)?,
            )
        } else {
            let mut qs = Vec::new();
            let mut ks = Vec::new();
            for row in 0..time {
                qs.push(
                    q.slice(2, row, row + 1)?
                        .rope(self.rope_dims, self.theta, base + row)?,
                );
                ks.push(
                    k.slice(2, row, row + 1)?
                        .rope(self.rope_dims, self.theta, base + row)?,
                );
            }
            (
                Array::concatenate(&qs.iter().collect::<Vec<_>>(), 2)?,
                Array::concatenate(&ks.iter().collect::<Vec<_>>(), 2)?,
            )
        };
        let v = qkv[2]
            .reshape(&[1, time, self.kv_heads, self.head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let keys = match &self.keys {
            Some(previous) => Array::concatenate(&[previous, &k], 2)?,
            None => k,
        };
        let values = match &self.values {
            Some(previous) => Array::concatenate(&[previous, &v], 2)?,
            None => v,
        };
        let mut attended = Vec::with_capacity(time as usize);
        for row in 0..time {
            let end = base + row + 1;
            let output = Array::sdpa(
                &q.slice(2, row, row + 1)?,
                &keys.slice(2, 0, end)?,
                &values.slice(2, 0, end)?,
                (self.head_dim as f32).powf(-0.5),
                false,
            )?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, 1, self.heads * self.head_dim])?
            .mul(&gate.slice(1, row, row + 1)?.sigmoid()?)?;
            attended.push(output);
        }
        let output = self.output.forward(&Array::concatenate(
            &attended.iter().collect::<Vec<_>>(),
            1,
        )?)?;
        self.keys = Some(keys);
        self.values = Some(values);
        Ok(output)
    }
}

#[test]
fn preparation_scope_restores_nested_and_panicking_overrides() {
    assert!(current().is_none());
    with_preparation(Some(Preparation::BatchAll), || {
        assert!(matches!(current(), Some(Preparation::BatchAll)));
        with_preparation(None, || assert!(current().is_none()));
        assert!(matches!(current(), Some(Preparation::BatchAll)));
        std::thread::spawn(|| assert!(current().is_none()))
            .join()
            .unwrap();
    });
    let _ = std::panic::catch_unwind(|| {
        with_preparation(Some(Preparation::ScalarRope), || panic!("scope test"));
    });
    assert!(current().is_none());
}

#[cfg(all(kani, test))]
#[kani::proof]
fn preparation_scope_restores_real_override() {
    let batch: bool = kani::any();
    with_preparation(
        Some(if batch {
            Preparation::BatchAll
        } else {
            Preparation::ScalarNormRope
        }),
        || {
            assert!(current().is_some());
            with_preparation(None, || assert!(current().is_none()));
            assert!(current().is_some());
        },
    );
    assert!(current().is_none());
    kani::cover!(batch);
    kani::cover!(!batch);
}

#[cfg(test)]
fn fixture(shape: &[i32], seed: u32) -> Result<Array> {
    let count: usize = shape.iter().map(|&n| n as usize).product();
    let bits = (0..count)
        .map(|i| {
            let word = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed);
            f16::from_f32(((word >> 8) % 1021) as f32 / 1021. - 0.5).to_bits()
        })
        .collect::<Vec<_>>();
    Array::from_f16_bits(&bits, shape)
}

#[cfg(test)]
fn checked_bytes(array: &Array) -> Result<Vec<u8>> {
    let bytes = array.to_bytes()?;
    ensure!(
        array.dtype() == Dtype::Float16 && !bytes.is_empty(),
        "empty/non-FP16 A1 output"
    );
    ensure!(
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| f16::from_bits(u16::from_ne_bytes([b[0], b[1]])).is_finite()),
        "non-finite A1 output"
    );
    Ok(bytes)
}

#[cfg(test)]
fn eval_states(model: &Qwen35Moe) -> Result<()> {
    let snapshot = model.snapshot()?;
    for array in profile::arrays(&snapshot) {
        array.eval()?;
    }
    Ok(())
}

#[cfg(test)]
fn candidate_verify(
    model: &mut Qwen35Moe,
    tokens: &[u32],
    production_route: bool,
) -> Result<(Array, Array)> {
    if production_route {
        let probe = Array::zeros_dtype(&[1, tokens.len() as i32, 5120], Dtype::Float16)?;
        let eligible = (2..=4).contains(&tokens.len())
            && model.offset() >= 16384
            && crate::array::is_m5_gpu()?;
        for layer in &model.layers {
            if let Layer::Attention(layer) = layer {
                ensure!(
                    selection(&layer.attention, &probe, tokens.len() as i32)?.is_some() == eligible,
                    "A1 production selection differs"
                );
            }
        }
        model.verify_mtp(tokens)
    } else {
        with_preparation(Some(Preparation::BatchAll), || model.verify_mtp(tokens))
    }
}

#[test]
#[ignore = "requires local Qwen checkpoint and physical Apple GPU"]
fn prepared_attention_exact_screen() -> Result<()> {
    let output = std::env::var("MLXL3_A1_REPORT")?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let checkpoint =
        crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_MTP_TEST_MODEL")?))?;
    let root: RootConfig =
        serde_json::from_reader(File::open(checkpoint.path.join("config.json"))?)?;
    let config = root.text_config;
    let layer = config
        .layer_types
        .iter()
        .position(|kind| kind == "full_attention")
        .context("no attention layer")?;
    let mut attention = Attention::load(
        &checkpoint,
        &format!("model.language_model.layers.{layer}.self_attn"),
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        (config.head_dim as f32 * config.partial_rotary_factor) as i32,
        config.rope_parameters.rope_theta,
        config.rms_norm_eps,
    )?;
    let mut cells = Vec::new();
    for past in [0, 1, 23, 257, 4096, 16384] {
        let initial = if past == 0 {
            None
        } else {
            Some((
                fixture(&[1, attention.kv_heads, past, attention.head_dim], 47)?,
                fixture(&[1, attention.kv_heads, past, attention.head_dim], 83)?,
            ))
        };
        for time in 1..=8 {
            let x = fixture(&[1, time, config.hidden_size], 101 + time as u32)?;
            let restore = |attention: &mut Attention| -> Result<()> {
                attention.keys = initial.as_ref().map(|(k, _)| k.try_clone()).transpose()?;
                attention.values = initial.as_ref().map(|(_, v)| v.try_clone()).transpose()?;
                attention.verification_base = None;
                Ok(())
            };
            restore(&mut attention)?;
            let expected =
                with_preparation(None, || attention.forward_verification_impl(&x, true))?;
            let bytes = checked_bytes(&expected)?;
            let expected_keys = checked_bytes(attention.keys.as_ref().unwrap())?;
            let expected_values = checked_bytes(attention.values.as_ref().unwrap())?;
            let full_keys = attention.keys.as_ref().unwrap().try_clone()?;
            let full_values = attention.values.as_ref().unwrap().try_clone()?;
            for mode in [
                Preparation::BatchAll,
                Preparation::ScalarRope,
                Preparation::ScalarNormRope,
            ] {
                restore(&mut attention)?;
                let actual =
                    with_preparation(Some(mode), || attention.forward_verification_impl(&x, true))?;
                let output_exact =
                    actual.shape() == expected.shape() && checked_bytes(&actual)? == bytes;
                let keys_exact = checked_bytes(attention.keys.as_ref().unwrap())? == expected_keys;
                let values_exact =
                    checked_bytes(attention.values.as_ref().unwrap())? == expected_values;
                let mut commits = true;
                let candidate_keys = attention.keys.as_ref().unwrap().try_clone()?;
                let candidate_values = attention.values.as_ref().unwrap().try_clone()?;
                for retained in 1..=time {
                    attention.keys = Some(candidate_keys.try_clone()?);
                    attention.values = Some(candidate_values.try_clone()?);
                    attention.verification_base = Some(past);
                    attention.commit_verification_prefix(retained, time)?;
                    commits &= checked_bytes(attention.keys.as_ref().unwrap())?
                        == checked_bytes(&full_keys.slice(2, 0, past + retained)?)?;
                    commits &= checked_bytes(attention.values.as_ref().unwrap())?
                        == checked_bytes(&full_values.slice(2, 0, past + retained)?)?;
                }
                cells.push(serde_json::json!({"past":past,"m":time,"mode":format!("{mode:?}"),"output_exact":output_exact,"keys_exact":keys_exact,"values_exact":values_exact,"all_commits_exact":commits}));
            }
            println!("A1 past={past} M={time}: screened three preparations");
        }
    }
    let report = serde_json::json!({"status":"complete","cells":cells,"boundary":"real attention weights, deterministic FP16 inputs; not full-model logits or throughput"});
    serde_json::to_writer_pretty(&mut file, &report)?;
    for mode in ["BatchAll", "ScalarRope", "ScalarNormRope"] {
        let matching = report["cells"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|cell| cell["mode"] == mode)
            .collect::<Vec<_>>();
        let exact = matching.iter().all(|cell| {
            [
                "output_exact",
                "keys_exact",
                "values_exact",
                "all_commits_exact",
            ]
            .iter()
            .all(|key| cell[*key] == true)
        });
        println!("A1 mode={mode} cases={} exact={exact}", matching.len());
    }
    Ok(())
}

#[test]
#[ignore = "requires local Qwen checkpoint and physical Apple GPU"]
fn prepared_attention_paired() -> Result<()> {
    use crate::array::synchronize;
    use std::time::Instant;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var("MLXL3_A1_REPORT")?)?;
    let checkpoint =
        crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_MTP_TEST_MODEL")?))?;
    let root: RootConfig =
        serde_json::from_reader(File::open(checkpoint.path.join("config.json"))?)?;
    let config = root.text_config;
    let layer = config
        .layer_types
        .iter()
        .position(|kind| kind == "full_attention")
        .context("no attention layer")?;
    let mut attention = Attention::load(
        &checkpoint,
        &format!("model.language_model.layers.{layer}.self_attn"),
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        (config.head_dim as f32 * config.partial_rotary_factor) as i32,
        config.rope_parameters.rope_theta,
        config.rms_norm_eps,
    )?;
    let mut cells = Vec::new();
    for past in [257, 4096, 16384, 32768] {
        let k = fixture(&[1, attention.kv_heads, past, attention.head_dim], 47)?;
        let v = fixture(&[1, attention.kv_heads, past, attention.head_dim], 83)?;
        k.eval()?;
        v.eval()?;
        for time in [2, 3, 4] {
            let x = fixture(&[1, time, config.hidden_size], 101 + time as u32)?;
            x.eval()?;
            let restore = |attention: &mut Attention| -> Result<()> {
                attention.keys = Some(k.try_clone()?);
                attention.values = Some(v.try_clone()?);
                attention.verification_base = None;
                Ok(())
            };
            let mut oracle = None;
            for mode in [None, Some(Preparation::BatchAll)] {
                restore(&mut attention)?;
                let output =
                    with_preparation(mode, || attention.forward_verification_impl(&x, true))?;
                let result = (
                    checked_bytes(&output)?,
                    checked_bytes(attention.keys.as_ref().unwrap())?,
                    checked_bytes(attention.values.as_ref().unwrap())?,
                );
                if let Some(expected) = &oracle {
                    ensure!(
                        *expected == result,
                        "A1 paired preflight divergence at past={past} M={time}"
                    );
                } else {
                    oracle = Some(result);
                }
            }
            let mut samples = [Vec::new(), Vec::new()];
            for repeat in 0..15 {
                for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                    restore(&mut attention)?;
                    synchronize()?;
                    let start = Instant::now();
                    let output =
                        with_preparation((arm == 1).then_some(Preparation::BatchAll), || {
                            attention.forward_verification_impl(&x, true)
                        })?;
                    output.eval()?;
                    attention.keys.as_ref().unwrap().eval()?;
                    attention.values.as_ref().unwrap().eval()?;
                    synchronize()?;
                    let seconds = start.elapsed().as_secs_f64();
                    if repeat >= 3 {
                        samples[arm].push(seconds);
                    }
                }
            }
            let output = with_preparation(Some(Preparation::BatchAll), || {
                restore(&mut attention)?;
                attention.forward_verification_impl(&x, true)
            })?;
            let result = (
                checked_bytes(&output)?,
                checked_bytes(attention.keys.as_ref().unwrap())?,
                checked_bytes(attention.values.as_ref().unwrap())?,
            );
            ensure!(
                Some(result) == oracle,
                "A1 paired final divergence at past={past} M={time}"
            );
            cells.push(serde_json::json!({"past":past,"m":time,"samples_seconds":samples,"bit_exact":true}));
            println!("A1 paired past={past} M={time} complete");
        }
    }
    serde_json::to_writer_pretty(
        &mut file,
        &serde_json::json!({"status":"complete","parity":true,"cells":cells,"boundary":"one real attention layer, host+GPU output/KV evaluated and synchronized; not model throughput"}),
    )?;
    Ok(())
}

#[test]
#[ignore = "requires local Qwen checkpoint and physical Apple GPU"]
fn prepared_model_verification_paired() -> Result<()> {
    use crate::array::{clear_cache, synchronize};
    use std::{
        io::{Seek, SeekFrom, Write},
        time::Instant,
    };
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var("MLXL3_A1_REPORT")?)?;
    let mut cells = Vec::new();
    let validation_only = std::env::var("MLXL3_A1_VALIDATION_ONLY").as_deref() == Ok("1");
    let production_route = std::env::var("MLXL3_A1_PRODUCTION_ROUTE").as_deref() == Ok("1");
    ensure!(
        !production_route
            || std::env::var("MLXL3_EXPERIMENTAL_BATCHED_VERIFY").as_deref() == Ok("1"),
        "production-route test requires explicit opt-in"
    );
    let write_report = |file: &mut File,
                        cells: &[serde_json::Value],
                        status: &str,
                        error: Option<String>|
     -> Result<()> {
        let bytes = serde_json::to_vec_pretty(
            &serde_json::json!({"status":status,"parity":status=="complete","cells":cells,"error":error,"boundary":"real target verify_mtp, stock/BatchAll; output+raw+full-state synchronized, no forced component fences; not MTP advance or sustained generation"}),
        )?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&bytes)?;
        file.set_len(bytes.len() as u64)?;
        file.sync_all()?;
        Ok(())
    };
    write_report(&mut file, &cells, "running", None)?;
    let result = (|| -> Result<()> {
        let mut model = Qwen35Moe::load(Path::new(&std::env::var("MLXL3_MTP_TEST_MODEL")?))?;
        ensure!(
            model.layers.len() == 64
                && model.mtp_layout.hidden_size == 5120
                && model.vocab == 248320,
            "A1 model test expects dense Qwen27B"
        );
        let contexts = std::env::var("MLXL3_A1_CONTEXTS")
            .unwrap_or_else(|_| "4096,16384".into())
            .split(',')
            .map(str::parse::<usize>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            !contexts.is_empty()
                && contexts.iter().all(|&n| (1..=32768).contains(&n))
                && contexts.windows(2).all(|pair| pair[0] < pair[1]),
            "invalid A1 contexts"
        );
        for context in contexts {
            while (model.offset() as usize) < context {
                let count = (context - model.offset() as usize).min(128);
                let tokens = (0..count)
                    .map(|i| 1 + ((model.offset() as usize + i) % 31) as u32)
                    .collect::<Vec<_>>();
                let (logits, raw) = model.forward_mtp(&tokens)?;
                logits.eval()?;
                raw.eval()?;
                eval_states(&model)?;
                synchronize()?;
                clear_cache()?;
            }
            let saved = model.snapshot()?;
            for time in [2, 3, 4] {
                let tokens = (1..=time as u32).collect::<Vec<_>>();
                model.restore(saved.clone())?;
                let (expected, expected_raw) =
                    with_preparation(None, || model.verify_mtp(&tokens))?;
                let logits = checked_bytes(&expected)?;
                let hidden = checked_bytes(&expected_raw)?;
                ensure!(
                    expected.shape().iter().product::<i32>() == time * 248320,
                    "incomplete A1 model logits"
                );
                let full = profile::StateReference::save(&model.snapshot()?)?;
                model.restore(saved.clone())?;
                let (actual, actual_raw) = candidate_verify(&mut model, &tokens, production_route)?;
                ensure!(
                    checked_bytes(&actual)? == logits && checked_bytes(&actual_raw)? == hidden,
                    "A1 model logits/hidden differ"
                );
                full.compare(&model.snapshot()?)?;
                drop(full);
                for retained in 1..=time {
                    model.restore(saved.clone())?;
                    let (_, raw) =
                        with_preparation(None, || model.verify_mtp(&tokens[..retained as usize]))?;
                    model.commit_dflash_verification(retained as usize, retained as usize)?;
                    model.set_mtp_hidden(raw.slice(1, retained - 1, retained)?)?;
                    let expected = profile::StateReference::save(&model.snapshot()?)?;
                    model.restore(saved.clone())?;
                    let (_, raw) = candidate_verify(&mut model, &tokens, production_route)?;
                    model.commit_dflash_verification(retained as usize, time as usize)?;
                    model.set_mtp_hidden(raw.slice(1, retained - 1, retained)?)?;
                    expected.compare(&model.snapshot()?)?;
                }
                let mut samples = [Vec::new(), Vec::new()];
                for repeat in 0..if validation_only { 0 } else { 11 } {
                    for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                        model.restore(saved.clone())?;
                        clear_cache()?;
                        synchronize()?;
                        let start = Instant::now();
                        let (output, raw) = if arm == 1 && production_route {
                            model.verify_mtp(&tokens)?
                        } else {
                            with_preparation((arm == 1).then_some(Preparation::BatchAll), || {
                                model.verify_mtp(&tokens)
                            })?
                        };
                        output.eval()?;
                        raw.eval()?;
                        eval_states(&model)?;
                        synchronize()?;
                        let seconds = start.elapsed().as_secs_f64();
                        if repeat >= 3 {
                            samples[arm].push(seconds);
                        }
                    }
                }
                model.restore(saved.clone())?;
                let (actual, raw) = candidate_verify(&mut model, &tokens, production_route)?;
                ensure!(
                    checked_bytes(&actual)? == logits && checked_bytes(&raw)? == hidden,
                    "A1 final logits/hidden differ"
                );
                cells.push(serde_json::json!({"context":context,"m":time,"samples_seconds":samples,"validation_only":validation_only,"production_route":production_route,"bit_exact":true,"state_arrays":129,"commits_checked":time}));
                write_report(&mut file, &cells, "running", None)?;
                println!(
                    "A1 model context={context} M={time}: full logits/129 states/all commits exact, validation_only={validation_only}"
                );
            }
            model.restore(saved)?;
        }
        Ok(())
    })();
    write_report(
        &mut file,
        &cells,
        if result.is_ok() { "complete" } else { "failed" },
        result.as_ref().err().map(ToString::to_string),
    )?;
    result
}
