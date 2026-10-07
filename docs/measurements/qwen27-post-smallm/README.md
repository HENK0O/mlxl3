# Preuves de la campagne post-small-M

Lire d'abord [le rapport](../../qwen27-m5-post-smallm.md) et `opti.md` à la racine. Aucun gain soutenu validé ; A1 OFF par défaut. Les JSON/logs négatifs et partiels sont conservés. Un statut `running` historique après arrêt n'est pas une campagne active : la commande associée et le journal expliquent son interruption.

## Mesures et identités

| Phase | Données / résumé | Provenance / limites |
|---|---|---|
| Checkpoint et matériel | [preflight.json](preflight.json), [conditions](conditions-observations.jsonl) |16fichiers cible/tête intégralement hashés ; pas de poids dansGit |
| Profil v2 réel | [matrice](profile-matrix-v2.json), [synthèse](profile-summary.json) | [commande](profile-matrix-v2-command.json), `profile-v2-*.txt` ; fences diagnostiques non additives |
| Profil v1 interrompu | [partiel](profile-matrix.json) | `profile-matrix-command.json`, `profile-v1-*` ; pression mémoire, pas de gain |
| A1 exact/composant | [144cas](a1-exact-screen.json), [timings](a1-attention-paired.json) | `a1-screen-*`, `a1-paired-*`, mutation `a1-causal-mutant-result.json` |
| A1 modèle batterie | [fenêtre1](a1-model-window1.json), [fenêtre2](a1-model-window2.json) | commandes/sources `a1-model-*` ; contrôles instables |
| A1 runtime actuel | [validation6cas](a1-runtime-recovery-validation.json), [commande](a1-runtime-recovery-validation-command.json) | `a1-runtime-*` ; variant stockcourt/longroute réel, exact |
| A1 secteur interrompu | [partiel](a1-ac-window1.json), [commande failed](a1-ac-window1-command.json) | deuxcas exacts, conditionsecteur échouée, pas de fenêtre2 |
| Bridge A1 | [contrôle](a1-bridge-quality.json), [commande](a1-bridge-quality-command.json) |16tokens/bras, aucun débit soutenu ; imports/borneprompt échecs séparés |
| E MB4 séparé | [fenêtre1](mb4-head-window1.json), [fenêtre2](mb4-head-window2.json) | `mb4-source-*`, stockshader ; ralentissement composant |
| E validation renforcée | [16projections complètes](mb4-full-projections-validation.json) | contrôle avanttanh, pas de nouveaux timings |
| A2/C | [312cas](a2c-screen.json), [synthèse](a2c-summary.json) | `a2c-command.json`, sourceMTPLX/licence épinglées ; parity=false |
| A2/C schema final | [audit312cas](a2c-validator-raw-audit.json) | pas de retiming, anciennes sources conservées |
| D adaptatif | [matrice](d-confidence-screen.json), [sweep offline](d-confidence-summary.json) | `d-confidence-screen-command.json`, `d-source-*` ;32casqualité/32tracerounds, prototypecfg(test) |
| B FP16 réserve | [matrice](b-reserved-kv-screen.json), [synthèse](b-reserved-kv-summary.json) | `b-source-*`, snapshots vivants ; composantinstable/sanssignal |
| G transforms | [audit48tensors](g-qkv-transforms-audit.json) |16couchestrunk, embeddedMTP exclu ; pas de partage simple |

Le code des benchmarks est dans `benchmarks/`; les harnais physiques Rust dans `native/src/qwen35/{profile,prepared}.rs` et `native/src/mtp/native/dynamic.rs`. Exemples de sélection explicite :

```sh
# Nécessite build release --features mlx,chat, checkpoint et GPU Apple.
MLXL3_MTP_TEST_MODEL=/chemin/model MLXL3_A1_REPORT=/chemin/nouveau.json cargo test --release --features mlx,chat --lib qwen35::prepared::prepared_attention_exact_screen -- --ignored --exact --nocapture --test-threads=1
# Mode de test du selector réel : ajouter MLXL3_A1_PRODUCTION_ROUTE=1,
# MLXL3_EXPERIMENTAL_BATCHED_VERIFY=1 et MLXL3_A1_VALIDATION_ONLY=1.
# Le harnais modèle utilise MLXL3_A1_CONTEXTS=24,16384.
```

Ne pas exécuter les harnais simultanément. Les chemins de sortie doivent être nouveaux ; ils ne remplacent pas les preuves précédentes. Pour D, ajouter MLXL3_MTP_TEST_HEAD et MLXL3_DYNAMIC_REPORT puis sélectionner `mtp::native::dynamic::real_adaptive_depth_screen`. D est exclusivement compilé en tests ; la variable A1 ne l'active pas.

## Vérification et limitations

Kani ciblé : `a1-runtime-proof-fixed.log`, `a1-runtime-kani-scope.log`, `d-policy-kani.log`, `d-scope-kani.log`. Contrôleurprofil source non vérifié : `profile-kani-source.log` (panique compilateur). CrossHair absent : diagnostics `*-crosshair.log`. Les obligations CPU ne prouvent pas Metal/MLX, concurrence ou qualité de toutes générations.

Contre-exemples/mutations : `a1-causal-mutant*`, `profile-scope-mutant*`, `d-policy-mutant.log`, `mb4-validator-mutant*`, `a2c-validator-red.log`, `a2c-signedzero-mutant.log`, `b-snapshot-mutant.log`. Sources restaurées avant vérifications finales. Les contrôles finaux et un manifeste SHA256 seront ajoutés à la clôture ; la CI distante du commit publié est indépendante des mesures physiques.

Les binaires figés restent dans le dossier local `../work/` horsGit : profilv2, A1runtime-tests/CLI et D-tests. Les commandJSON conservent leurs hashes. `runtime_commit` bridge signale base+dirty parce que les binaires ont été compilés avant commit ; les empreintes et archives identifient leurs sources effectives.
