# Qwen3.8-27B EXL3 / M5 Air — campagne post-small-M

**Aucun gain soutenu en tokens/s n'est validé.** Deux options expérimentales sont désactivées par défaut : A1 regroupe la préparation d'attention avec des résultats exacts dans les contrôles ; H rend environ **3,99 Go après un prompt de 16 710 tokens**, confirmé dans les deux ordres de comparaison. Les autres expériences restent dans les outils de test, ont échoué au filtre de promotion ou n'ont pas été déclenchées. Aucune app installée ni aucun poids n'a été modifié par cette campagne.

Branche : `optimize/qwen27-post-smallm`. Campagne du 7 au 8 octobre 2026. Les mesures initiales portent sur le moteur 1.4.1, base `8f63e6c89e656bb1e582d9a3dc92b1135b275de0`, avec MLX 0.32.2 sur M5 Air / 10 cœurs GPU / 24 GiB / macOS 27.2. Main a ensuite avancé vers **1.4.2**, `d20ca721d808759f07ecf8976274d878e64a1ed3`, intégré à cette branche. Les timings historiques ne valident pas un gain sur ce nouveau moteur. Les contrôles de correction affectés sont répétés séparément sous `sync142-*` ; les améliorations MTP et le réglage de déchargement déjà intégrés dans main sont conservés.

## Décisions

| Candidat | Correction observée | Composant | Verify / rounds natifs | Génération réelle | Mémoire | Décision |
|---|---|---|---|---|---|---|
| A1 normes/RoPE batchés, concatKV unique | 144 cas attention / 648 commits exacts ; 6 cas modèle sur 1.4.2 exacts, full-logits/hidden/129 états/commits | Signal long, contrôle instable | Deux fenêtres batterie non concluantes ; vrai selector vérifié | Bridge 1.4.1 puis 1.4.2 exacts, prompts long/court/cache | Gain A1 seul non mesuré | Expérimental OFF |
| A2 un tail-causal SDPA stock | 36/96cas changent des bits | Non chronométré après échec exact | Non exécuté | Non exécutée | Non mesurée | Rejeté pour contrat exact |
| B réserve FP16 fonctionnelle | 9cas/27préfixes/forks/snapshots complets exacts | −10,89..+0,84%, contrôle instable ; filtre8% échoué | Non exécuté | Non exécutée | Compteurs MLX agrégés, pas de gain attribué | Variante rejetée ; cache mutable versionné non développé |
| C MTPLX NAX | 0/216cas bitexact ; maxdiff fixture≤3,052e−5 | Plafond exploratoire seulement | Logits/argmax/acceptance modèle non vérifiés | Non exécutée | Non mesurée | Hors runtime, contrat numérique distinct |
| D profondeur adaptative top1 | 32comparaisons full-logits/129états/cache/IDs exactes | Score sans softmax supplémentaire | 12formes×8paires ; sweep25seuils/workload sans signal3% | Non exécutée | Gain non mesuré | Prototype cfg(test), pas de politique promue |
| E lm_head vrai MB4/NT1 | PartialsFP32/forwardFP16/16projections complètes exacts | Isolé−7,23/−5,02%, dépendant−6,58/−6,21% | Non déclenché après échec composant | Non exécutée | Non mesurée | Rejeté, aucun nouveau dispatch natif |
| F KV INT8 | Non exécuté | Non mesuré | Non mesuré | Non exécutée | Non mesurée | Secondaire après cache exact utile, non déclenché |
| G préparation QKV hétérogène | Audit48transforms finies ; aucune paire identique sur16couches | Préparation seule non isolée | Non exécuté | Non exécutée | Non mesurée | Fusion complexe non déclenchée |
| H libération du scratch de préfill MTP long | Garde CPU, primitive allocateur et bridge 1.4.2 exacts | Vitesse soutenue non mesurée | Aucun flush dans le decode | 15 requêtes / 240 tokens, court froid/réutilisé exact | Après long −3,986 / −3,985 Go ; pic proche du stock | Mémoire observée confirmée, expérimental OFF |

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

**H** : après un prompt réel de 23 470 tokens, le bridge historique retenait environ 6,90 Go d'allocations MLX inutilisées, en plus de 11,20 Go actifs. Le prototype `MLXL3_EXPERIMENTAL_PREFILL_CACHE_RELEASE=1` libère ce scratch après les chunks évalués, tous les 4 096 tokens et en fin de préfill MTP, uniquement pour les prompts d'au moins 16 384 tokens. Les poids et tableaux vivants restent référencés ; les snapshots du cache de prompt sont capturés avant la libération. Aucun changement de précision ou de limite mémoire, aucune libération dans le decode. L'option est désactivée par défaut et sa garde utilise des bornes vérifiées par Kani.

Le premier contrôle 1.4.2 a rejeté le protocole de réutilisation longue : la limite existante de 256 Mio par snapshot empêche déjà de conserver les KV de ce dense à 16k. Baseline seule exécutée, deux réponses complètes mais zéro token réutilisé ; échec conservé après 519,22 s, aucun effet H mesuré. Le protocole corrigé compare A1=0/H=0, A1=0/H=1 et A1=1/H=1 avec une requête longue froide (16 710 tokens), puis un prompt court froid et réutilisé (1 110 tokens dont 1 024 réutilisés). La borne mémoire reste intacte. Le driver reproductible `benchmarks/benchmark_prefill_release.py` garde deux oracles distincts par prompt, des délais bornés et des rapports d'échec persistés. Les 18 tests du driver et une mutation supprimant l'exigence de réutilisation passent après restauration.

V2 et confirmation H→stock sont complètes : **15 requêtes / 240 tokens**, mêmes IDs, texte, historique et compteurs MTP pour chaque prompt, cinq processus fermés/rejoints. [Synthèse mémoire](measurements/qwen27-post-smallm/h-memory-summary.json) :

| Ordre / contexte long | Stock (Go) | H seul (Go) | Mémoire rendue |
|---|---:|---:|---:|
| Stock→H | 18,2805 | 14,2941 | 3,9864 Go / 3,713 GiB |
| H→stock, processus neufs | 18,2800 | 14,2953 | 3,9848 Go / 3,711 GiB |

Ce sont des footprints processus relevés après la réponse. Le cache allocateur inutilisé passe d'environ 7,384 à 3,399 Go ; actif MLX environ 10,731 Go, différence de 16 KiB dans la seconde fenêtre. Les pics restent proches du stock. Après le court/cache, environ 2,14 Go restent rendus. A1+H donne 14,024 Go après le long dans V2, observation combinée à ne pas additionner aux économies de H seul. Alimentation batterie puis transition secteur lors de la confirmation : ces requêtes de 16 tokens ne valident aucune accélération soutenue. H reste OFF tant que l'impact sur latence/débit n'est pas établi.

Pour expérimenter H dans un nouveau processus moteur : `MLXL3_EXPERIMENTAL_PREFILL_CACHE_RELEASE=1`. Le premier audit de synthèse exigeait à tort des compteurs d'allocation identiques entre processus ; échec de 16 KiB conservé puis correction du post-traitement, sans changer les données ni la parité des résultats. Aucune nouvelle mesure pour cette correction.

**E** : quatre vraies lignes5120→248320/K3/CB2/SG8/split1, MB2×2/NT2 contre MB4×1/NT1, aucun padding/groupedMB3. Oracle quatreM1/NT1, partialsFP32/transforms/scalesFP16 et huit forwards dépendants ;40paires/fenêtre et3warmups. Isolé15,47→16,72ms puis16,27→17,12ms. Aucune cause de spill inventée. La validation renforcée vérifie les16projections complètes avant tanh, qui pourrait masquer Inf dans le feedback. Filtre composant échoué : pas de nativeverify/E2E/NT2 spéculatif.

**A2/C** :96fixtures tail stock,60exactes/36divergentes. C importe les shaders upstream inchangés, référence Apache2.0 `9882703f3105363ddc37eca9f97aa09a1d387112`/release2.12.2. M2/3/4×past8k/16k/32k×KS1/2/4/8×blocks32/64/128×Qstaging0/1,3warmups/12paires. Overrides de buffers refusés ; module upstream conserve les defaults quand overrides absents.216cas supportés, tous finis et sous la tolérance de fixture0,005, aucun bitexact. Plafonds composant ne garantissent pas l'intelligence : full-logits/argmax/acceptance/qualité utilisateur non vérifiés. Aucun nouveau kernel NAX dans le moteur. Les résultats Max MTPLX ne sont pas repris comme gains Air.

**D** : conserve `log_probs().argmax()` et extrait le logit brut du gagnant ; transfertID+scoreFP32 par étape, comparé au stock lazy à transfert unique. D1 obligatoire ; T1/T2 peuvent déclencherD2/D3, largeur effective pour verify/commit/proposed. Budgets context/output physiquement testés ; scopes restaurés après erreur/panic et isolés par thread.4workloads×8rounds observés ; sweep25seuils choisit toujoursD1. Gains du modèle de coûts offline contre meilleurfixe+0,05/−0,02/−4,05/−0,25% : aucun signal3%, plusieurs contrôles instables, pas un débit soutenu. Pas de runtime adaptatif/retuneapp/E2E inutile. Top2margin non essayé faute d'extraction gratuite démontrée.

**B/F** : réserve fonctionnelle slice_update past+1024 avec anciens snapshots vivants, pas de donation forcée ni cachemutable partagé. KV/attention/préfixes et forksrollback exacts, y compris fenêtre future complète. Aucun signal8% ; compteurs MLX agrégés sur les deux bras, pas de gain attribuable. Footprint processus B non mesuré. Réserve mutable versionnée : autre architecture non développée. INT8KV reste secondaire et non exécuté, contrat qualité distinct.

**G** :48suh FP16 finies aux formes originales, toutes pairesQ/K/V différentes sur16couches. Pas de Hadamard commun simple à réutiliser. Exl3Group batch déjà les préparations compatibles. Coût de préparation hétérogène suffisant non isolé : pas de grand kernel complexe/requantification.

## Vérification

Kani0.68/CBMC6.11, vraies fonctions et entrées symboliques : gardeA1 forme/M5/offset70obligations/3covers ; policyD budget/NaN/Inf/seuils69obligations/4covers ; DropTLS D76obligations/1cover. Model checks CPU, pas preuves de MLX/Metal/allocateur/intelligence. ContrôleurRefCell du profil : tentative échouée par panique du compilateur ARM, non vérifié formellement. CrossHair absent localement : diagnostics conservés, aucune preuve Python revendiquée.

Mutations détectées/restaurées : fuitecausaleA1, scoreInfD, finitude/timingMB4 et égaliténumérique remplaçant octets/signedzero A2/C/B. Comparateur A2/C initial accepte mal duplicatas/métriquesinvalides :4contre-exemples rouges, schema corrigé,312cellules brutes revalidées sans retiming. Une divergence/interruption ne devient jamais parity=true. Erreurs/timeout des relevés système sont explicites et ne masquent pas une divergence originale.

Sur le code intégré à 1.4.2 : format, Clippy strict et builds release MLX/chat réussis ; 75 tests Rust passés, 65 ignorés ; les 42 harnais Kani CPU passent. La suite Python native initiale passe 337 tests ; après ajout du driver H, la suite complète passe 357 tests et en saute 4 dans l'environnement CI jetable. Les parcours Desktop passent avec SDK26.5, dont 114 assertions idle-unload. Le premier échec de collecte faute de dépendance est conservé. Les tests physiques sont ignorés dans la CI hébergée et sélectionnés explicitement sur ce Mac : primitive allocateur réellement exécutée, six cas A1 sous la route de production exacts sur 1.4.2 en 226,05 s diagnostiques, puis 32 cas D exacts en 265,69 s. Aucun timing de performance relancé dans ces contrôles de correction. Les runs CI du SHA publié sont inspectés séparément après publication ; leurs captures locales restent hors des mesures physiques. CrossHair demeure absent, sans preuve Python revendiquée.

## Provenance et reproduction

[Dossier des preuves](measurements/qwen27-post-smallm/README.md) : commandes/options/environnement, checkpoint, hashes binaires, sources figées par phase, conditions et échecs.16fichiers cible/tête rehashés intégralement, identiques à la campagne précédente. Sources de mesure historiques distinctes des validateurs finals ; binaires volumineux horsGit mais empreintes conservées. Un seul modèle GPU, watchdogs/enfants rejoints.

Gates : composant8–10%, nativeverify3% sur deuxfenêtres, E2E3% sur256+tokens (préférence512) ABBA/BAAB, ou long5%/courtneutre ; contrôle≤5% et conditions comparables. Aucun candidat ne franchit cette chaîne de promotion en vitesse. Cette branche permet de revoir/tester A1 et H explicitement ; elle ne promet pas davantage de tokens/s avec les réglages actuels. Tous les essais lancés sont clos ; la CI distante du commit publié reste une vérification distincte.
