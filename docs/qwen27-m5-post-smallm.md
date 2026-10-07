# Qwen3.8-27B EXL3 / M5 Air — campagne post-small-M

**Aucun gain soutenu en tokens/s n'est validé.** A1 est une préparation d'attention exacte disponible sous opt-in explicite, désactivée par défaut. Les autres expériences restent dans les outils de test, ont échoué au filtre de promotion ou n'ont pas été déclenchées. L'app installée et les poids sont inchangés.

Base upstream main : `8f63e6c89e656bb1e582d9a3dc92b1135b275de0`. Branche : `optimize/qwen27-post-smallm`. Mesures du7octobre2026 sur M5Air/10GPU/24GiB/macOS27.2, MLX0.32.2, source moteur1.4.1. Le texte de départ mentionne1.4.2 : aucune version1.4.2 trouvée dans la base récupérée. Les essais small-M antérieurs restent distincts de cette campagne.

## Décisions

| Candidat | Correction observée | Composant | Verify / rounds natifs | Génération réelle | Mémoire | Décision |
|---|---|---|---|---|---|---|
| A1 normes/RoPE batchés, concatKV unique | 144cas attention/648commits exacts ; full-logits/hidden/129états/commits exacts | Signal long, contrôle instable | Deux fenêtres batterie nonconcluantes ; vrai selector vérifié | Contrôle bridge16tokens/bras en cours, pas256+tokens de vitesse | Gain non mesuré | Expérimental OFF |
| A2 un tail-causal SDPA stock | 36/96cas changent des bits | Non chronométré après échec exact | Non exécuté | Non exécutée | Non mesurée | Rejeté pour contrat exact |
| B réserve FP16 fonctionnelle | 9cas/27préfixes/forks/snapshots complets exacts | −10,89..+0,84%, contrôle instable ; filtre8% échoué | Non exécuté | Non exécutée | Compteurs MLX agrégés, pas de gain attribué | Variante rejetée ; cache mutable versionné non développé |
| C MTPLX NAX | 0/216cas bitexact ; maxdiff fixture≤3,052e−5 | Plafond exploratoire seulement | Logits/argmax/acceptance modèle non vérifiés | Non exécutée | Non mesurée | Hors runtime, contrat numérique distinct |
| D profondeur adaptative top1 | 32comparaisons full-logits/129états/cache/IDs exactes | Score sans softmax supplémentaire | 12formes×8paires ; sweep25seuils/workload sans signal3% | Non exécutée | Gain non mesuré | Prototype cfg(test), pas de politique promue |
| E lm_head vrai MB4/NT1 | PartialsFP32/forwardFP16/16projections complètes exacts | Isolé−7,23/−5,02%, dépendant−6,58/−6,21% | Non déclenché après échec composant | Non exécutée | Non mesurée | Rejeté, aucun nouveau dispatch natif |
| F KV INT8 | Non exécuté | Non mesuré | Non mesuré | Non exécutée | Non mesurée | Secondaire après cache exact utile, non déclenché |
| G préparation QKV hétérogène | Audit48transforms finies ; aucune paire identique sur16couches | Préparation seule non isolée | Non exécuté | Non exécutée | Non mesurée | Fusion complexe non déclenchée |

Les pourcentages de composant sont des ratios de débit inverse du temps, pas des gains à additionner. Une dérive>5% ne valide aucun gain, même avec une médiane favorable. Pas de requantification, nouveau checkpoint, hausse de limites MLX ou réglage d'alimentation.

## Profil réel et limites

[Profil et synthèse](measurements/qwen27-post-smallm/profile-summary.json) :24cellules code/FR×4096/8192/16384/32768×D1/2/3,216rounds mesurés après warmup,1342,02s. Normal/graph/fenced conservent tokens, full-logits,129états à leur dtype original, cache draft et compteurs. Préfixes tokenisés répétés : diagnostic, pas benchmark de chat soutenu représentatif.

Les16fullattention/48GDN/64MLP passent par les vrais opérateurs. Les catégories QKV/normes/RoPE/concat/SDPA/gate/o_proj sont séparées. Clôturer chaque sous-opérateur modifie la planification : durées wall hôte+GPU, **ni additives ni temps GPU purs**. À32k/code, concat fenced médian D1/2/3≈281/240/375ms ; cela motive A1/B, sans prouver une économie égale dans le round normal. Active/cache/peak MLX et footprint processus sont conservés par bras. Registres, spills, occupation et fréquences GPU non mesurés.

La première matrice a été interrompue après6cellules exactes : référencesGPU et conversions temporaires causaient une pression allocateur/swap global1690→8425MiB. V2 compare les octets originaux CPU, sauvegarde les références dans des fichiers temporaires supprimés à la sortie et libère seulement le cache allocateur inutilisé hors timer. Les états GDN FP32 ne sont jamais comparés après castFP16. Sources et échecs v1 conservés. Secteur/batterie fluctuants et contrôles normaux souvent>5% : attribution diagnostique seulement.

## A1 : chemin exact pour expérimentation

[Implémentation](../native/src/qwen35/prepared.rs), [garde CPU](../native/src/contracts.rs) : QKV stock, normes/RoPE batchés, un appendK et un appendV. Chaque ligne garde exactement le SDPA qlen1 sur `past+row+1`, sa gate et l'o_proj stock. Aucun futur token visible ; `verification_base` et commit/rollback existants conservés.

Avant de lancer un nouveau processus moteur :

```sh
MLXL3_EXPERIMENTAL_BATCHED_VERIFY=1 /chemin/vers/mlxl3-rs …
```

Garde : M5/hidden5120/heads24/kv4/d256/M2..4/past≥16384 et addition sans overflow. Les autres cas gardent stock. Variable absente ou différente de `1` : OFF. Lu une fois par processus ; pour désactiver, retirer la variable et relancer le moteur. Ce n'est pas un setting Desktop et la branche n'installe pas l'app.

Trois variantes ont été filtrées sur les vrais poids/normes, M1..8 et past0/1/23/257/4096/16384. Mutation donnant à chaque ligne la fenêtre future complète :126/144cas divergents,18contrôlesM1 exacts. L'oracle détecte une fuite future réelle. BatchAll est la seule variante runtime.

Verify complet sans fences : full-logits/hidden/129états/tous commits exacts, timer inclut verify+évaluation+synchronisation :

| Contexte / fenêtre | M2 gain apparié | M3 | M4 | Dérive contrôle A M2/M3/M4 |
|---|---:|---:|---:|---|
| 4096 / batterie1 | +1,92% | +1,46% | −1,14% | 3,11/6,38/24,27% |
| 16384 / batterie1 | +5,09% | +7,31% | +4,39% | 9,26/16,98/18,77% |
| 16384 / batterie2 | +7,46% | +3,00% | +4,27% | 19,55/12,93/21,04% |

À16k/fenêtre1, médianes stock→candidat M2/3/4 :417,63→396,88 /686,92→642,26 /842,24→809,36ms. Aucun gain soutenu démontré. Une tentative sous secteur observé a été interrompue ; reprise : vrai selector exact sur6cellules past24/16384×M2..4,194,79s. Les conditions effectives étaient déjà batterie. AC1 arrêtée/rejointe après280,56s avec deux cellules exactes, rapport partiel parity=false ; AC2 non exécutée. Pas de gate abaissé ou nouvelle relance longue pour promouvoir A1.

## Filtres négatifs et pistes conditionnelles

**E** : quatre vraies lignes5120→248320/K3/CB2/SG8/split1, MB2×2/NT2 contre MB4×1/NT1, aucun padding/groupedMB3. Oracle quatreM1/NT1, partialsFP32/transforms/scalesFP16 et huit forwards dépendants ;40paires/fenêtre et3warmups. Isolé15,47→16,72ms puis16,27→17,12ms. Aucune cause de spill inventée. La validation renforcée vérifie les16projections complètes avant tanh, qui pourrait masquer Inf dans le feedback. Filtre composant échoué : pas de nativeverify/E2E/NT2 spéculatif.

**A2/C** :96fixtures tail stock,60exactes/36divergentes. C importe les shaders upstream inchangés, référence Apache2.0 `9882703f3105363ddc37eca9f97aa09a1d387112`/release2.12.2. M2/3/4×past8k/16k/32k×KS1/2/4/8×blocks32/64/128×Qstaging0/1,3warmups/12paires. Overrides de buffers refusés ; module upstream conserve les defaults quand overrides absents.216cas supportés, tous finis et sous la tolérance de fixture0,005, aucun bitexact. Plafonds composant ne garantissent pas l'intelligence : full-logits/argmax/acceptance/qualité utilisateur non vérifiés. Aucun nouveau kernel NAX dans le moteur. Les résultats Max MTPLX ne sont pas repris comme gains Air.

**D** : conserve `log_probs().argmax()` et extrait le logit brut du gagnant ; transfertID+scoreFP32 par étape, comparé au stock lazy à transfert unique. D1 obligatoire ; T1/T2 peuvent déclencherD2/D3, largeur effective pour verify/commit/proposed. Budgets context/output physiquement testés ; scopes restaurés après erreur/panic et isolés par thread.4workloads×8rounds observés ; sweep25seuils choisit toujoursD1. Gains du modèle de coûts offline contre meilleurfixe+0,05/−0,02/−4,05/−0,25% : aucun signal3%, plusieurs contrôles instables, pas un débit soutenu. Pas de runtime adaptatif/retuneapp/E2E inutile. Top2margin non essayé faute d'extraction gratuite démontrée.

**B/F** : réserve fonctionnelle slice_update past+1024 avec anciens snapshots vivants, pas de donation forcée ni cachemutable partagé. KV/attention/préfixes et forksrollback exacts, y compris fenêtre future complète. Aucun signal8% ; compteurs MLX agrégés sur les deux bras, pas de gain attribuable. Footprint processus B non mesuré. Réserve mutable versionnée : autre architecture non développée. INT8KV reste secondaire et non exécuté, contrat qualité distinct.

**G** :48suh FP16 finies aux formes originales, toutes pairesQ/K/V différentes sur16couches. Pas de Hadamard commun simple à réutiliser. Exl3Group batch déjà les préparations compatibles. Coût de préparation hétérogène suffisant non isolé : pas de grand kernel complexe/requantification.

## Vérification

Kani0.68/CBMC6.11, vraies fonctions et entrées symboliques : gardeA1 forme/M5/offset70obligations/3covers ; policyD budget/NaN/Inf/seuils69obligations/4covers ; DropTLS D76obligations/1cover. Model checks CPU, pas preuves de MLX/Metal/allocateur/intelligence. ContrôleurRefCell du profil : tentative échouée par panique du compilateur ARM, non vérifié formellement. CrossHair absent localement : diagnostics conservés, aucune preuve Python revendiquée.

Mutations détectées/restaurées : fuitecausaleA1, scoreInfD, finitude/timingMB4 et égaliténumérique remplaçant octets/signedzero A2/C/B. Comparateur A2/C initial accepte mal duplicatas/métriquesinvalides :4contre-exemples rouges, schema corrigé,312cellules brutes revalidées sans retiming. Une divergence/interruption ne devient jamais parity=true. Erreurs/timeout des relevés système sont explicites et ne masquent pas une divergence originale.

Les tests physiques sont ignorés dans la CI hébergée et sélectionnés explicitement sur ce Mac. Format/lint/compilation/tests Python et contrats CPU sont ajoutés au workflow existant. Validation locale et CI du SHA publié seront enregistrées séparément à la clôture.

## Provenance et reproduction

[Dossier des preuves](measurements/qwen27-post-smallm/README.md) : commandes/options/environnement, checkpoint, hashes binaires, sources figées par phase, conditions et échecs.16fichiers cible/tête rehashés intégralement, identiques à la campagne précédente. Sources de mesure historiques distinctes des validateurs finals ; binaires volumineux horsGit mais empreintes conservées. Un seul modèle GPU, watchdogs/enfants rejoints.

Gates : composant8–10%, nativeverify3% sur deuxfenêtres, E2E3% sur256+tokens (préférence512) ABBA/BAAB, ou long5%/courtneutre ; contrôle≤5% et conditions comparables. Aucun candidat ne franchit cette chaîne. Cette branche permet de revoir/tester A1 explicitement ; elle ne promet pas davantage de tokens/s avec les réglages actuels.
