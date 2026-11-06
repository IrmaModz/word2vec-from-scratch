# Experiment results

Tokens: 131631, vocabulary: 6982

Baseline: D=50, LEARNING_RATE=0.025, EPOCHS=5, NEG_SAMPLES=2, WINDOW_SIZE=2

## Runs

| Run | D | lr | epochs | k | window | time (s) | loss | related sim | random sim |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 50 | 0.025 | 5 | 2 | 2 | 80.1 | 1.7141 | 0.684 | 0.445 |
| D=100 | 100 | 0.025 | 5 | 2 | 2 | 88.7 | 1.7125 | 0.677 | 0.449 |
| D=200 | 200 | 0.025 | 5 | 2 | 2 | 108.5 | 1.7146 | 0.687 | 0.454 |
| learning_rate=0.005 | 50 | 0.005 | 5 | 2 | 2 | 77.5 | 1.8606 | 0.908 | 0.502 |
| learning_rate=0.1 | 50 | 0.1 | 5 | 2 | 2 | 77.3 | 1.7185 | 0.475 | 0.204 |
| epochs=1 | 50 | 0.025 | 1 | 2 | 2 | 17.1 | 1.8704 | 0.884 | 0.429 |
| epochs=15 | 50 | 0.025 | 15 | 2 | 2 | 234.7 | 1.5907 | 0.594 | 0.279 |
| neg_samples=5 | 50 | 0.025 | 5 | 5 | 2 | 92.2 | 2.4105 | 0.714 | 0.692 |
| window_size=5 | 50 | 0.025 | 5 | 2 | 5 | 233.7 | 1.8234 | 0.595 | 0.482 |

## Cosine similarity of related pairs

| Pair | baseline | D=100 | D=200 | learning_rate=0.005 | learning_rate=0.1 | epochs=1 | epochs=15 | neg_samples=5 | window_size=5 |
|---|---|---|---|---|---|---|---|---|---|
| he/she | 0.710 | 0.685 | 0.701 | 0.956 | 0.459 | 0.947 | 0.672 | 0.804 | 0.502 |
| his/her | 0.588 | 0.584 | 0.597 | 0.911 | 0.523 | 0.919 | 0.615 | 0.604 | 0.490 |
| mr/mrs | 0.649 | 0.630 | 0.643 | 0.876 | 0.474 | 0.817 | 0.510 | 0.599 | 0.497 |
| mother/father | 0.722 | 0.736 | 0.745 | 0.938 | 0.730 | 0.816 | 0.619 | 0.758 | 0.612 |
| sister/brother | 0.776 | 0.755 | 0.760 | 0.914 | 0.539 | 0.884 | 0.643 | 0.761 | 0.613 |
| daughter/son | 0.381 | 0.383 | 0.384 | 0.849 | 0.241 | 0.896 | 0.450 | 0.520 | 0.524 |
| morning/evening | 0.782 | 0.789 | 0.792 | 0.961 | 0.345 | 0.964 | 0.616 | 0.839 | 0.709 |
| day/night | 0.614 | 0.609 | 0.619 | 0.809 | 0.318 | 0.588 | 0.539 | 0.709 | 0.588 |
| two/three | 0.795 | 0.762 | 0.785 | 0.958 | 0.590 | 0.971 | 0.641 | 0.715 | 0.716 |
| three/four | 0.873 | 0.866 | 0.871 | 0.922 | 0.655 | 0.971 | 0.653 | 0.833 | 0.743 |
| man/woman | 0.773 | 0.767 | 0.772 | 0.927 | 0.501 | 0.918 | 0.650 | 0.757 | 0.632 |
| lady/gentleman | 0.541 | 0.558 | 0.578 | 0.872 | 0.325 | 0.912 | 0.514 | 0.664 | 0.510 |

## Nearest neighbors (K=8)

### she

| Run | Neighbors | Shared with baseline |
|---|---|---|
| baseline | he (0.71), mary (0.64), disheartened (0.63), downwards (0.61), proudly (0.61), signify (0.60), perceive (0.60), unjustly (0.59) | 8/8 |
| D=100 | he (0.69), mary (0.63), disheartened (0.61), proudly (0.60), downwards (0.59), perceive (0.58), nicholls (0.58), charlotte (0.58) | 6/8 |
| D=200 | he (0.70), mary (0.64), disheartened (0.62), proudly (0.61), downwards (0.59), perceive (0.59), contain (0.58), nicholls (0.58) | 6/8 |
| learning_rate=0.005 | he (0.96), elizabeth (0.83), they (0.73), pronounce (0.71), perceive (0.69), yet (0.69), exactly (0.67), long (0.67) | 2/8 |
| learning_rate=0.1 | secretly (0.61), elizabeth (0.59), elevate (0.57), pronounce (0.57), increased (0.56), composition (0.55), gaudy (0.55), disconcerted (0.55) | 0/8 |
| epochs=1 | he (0.95), elizabeth (0.85), they (0.78), ambition (0.73), afford (0.72), jane (0.71), perceive (0.71), unwelcome (0.71) | 2/8 |
| epochs=15 | he (0.67), gracefully (0.63), unjustly (0.58), recollecting (0.56), rejected (0.55), nicholls (0.55), unwell (0.55), disheartened (0.54) | 3/8 |
| neg_samples=5 | he (0.80), mary (0.75), elizabeth (0.67), charlotte (0.66), recollecting (0.64), positively (0.64), finally (0.63), doubted (0.63) | 2/8 |
| window_size=5 | herself (0.61), certainty (0.61), appease (0.59), affectionately (0.58), reflected (0.58), recollecting (0.58), fluently (0.58), open (0.57) | 0/8 |

### mother

| Run | Neighbors | Shared with baseline |
|---|---|---|
| baseline | sister (0.81), ladyship (0.78), uncle (0.76), niece (0.76), brother (0.76), father (0.72), husband (0.71), heart (0.71) | 8/8 |
| D=100 | sister (0.80), uncle (0.79), niece (0.77), ladyship (0.76), brother (0.76), father (0.74), hair (0.72), husband (0.71) | 7/8 |
| D=200 | sister (0.81), uncle (0.79), niece (0.77), brother (0.77), ladyship (0.77), father (0.74), correspondence (0.72), hair (0.71) | 6/8 |
| learning_rate=0.005 | sister (0.97), ladyship (0.96), uncle (0.95), brother (0.95), father (0.94), friend (0.93), thoughts (0.92), husband (0.92) | 6/8 |
| learning_rate=0.1 | father (0.73), darting (0.69), rapacity (0.68), grandfather (0.66), infliction (0.64), sister (0.62), husband (0.62), superintends (0.60) | 3/8 |
| epochs=1 | sister (0.93), brother (0.93), intelligence (0.92), thoughts (0.91), carriage (0.91), eyes (0.91), ladyship (0.89), kindness (0.89) | 3/8 |
| epochs=15 | niece (0.67), husband (0.66), uncle (0.64), sister (0.63), father (0.62), neighbours (0.57), darting (0.56), parents (0.56) | 5/8 |
| neg_samples=5 | uncle (0.81), niece (0.80), ladyship (0.78), heart (0.77), husband (0.76), sister (0.76), father (0.76), maria (0.76) | 7/8 |
| window_size=5 | ladyship (0.76), sash (0.65), nieces (0.65), niece (0.64), husband (0.64), omen (0.63), gown (0.62), running (0.62) | 3/8 |

### sister

| Run | Neighbors | Shared with baseline |
|---|---|---|
| baseline | cousin (0.82), mother (0.81), niece (0.79), ladyship (0.78), brother (0.78), uncle (0.77), father (0.77), aunt (0.73) | 8/8 |
| D=100 | mother (0.80), cousin (0.79), niece (0.78), ladyship (0.78), uncle (0.77), brother (0.76), father (0.75), husband (0.73) | 7/8 |
| D=200 | cousin (0.81), mother (0.81), niece (0.79), ladyship (0.77), uncle (0.77), brother (0.76), father (0.75), aunt (0.73) | 8/8 |
| learning_rate=0.005 | mother (0.97), friend (0.94), ladyship (0.92), father (0.92), brother (0.91), uncle (0.91), own (0.90), cousin (0.88) | 6/8 |
| learning_rate=0.1 | grandfather (0.68), errand (0.65), correspondence (0.65), witticisms (0.64), inadequate (0.64), face (0.62), talks (0.62), mother (0.62) | 1/8 |
| epochs=1 | mother (0.93), daughter (0.92), ladyship (0.91), eyes (0.90), behaviour (0.89), uncle (0.89), friend (0.89), aunt (0.89) | 4/8 |
| epochs=15 | brother (0.64), ladyship (0.64), mother (0.63), eldest (0.62), fitzwilliam (0.58), steward (0.57), resignation (0.57), sorrow (0.55) | 3/8 |
| neg_samples=5 | ladyship (0.81), cousin (0.80), heart (0.80), commendation (0.79), patroness (0.79), nephew (0.79), aunt (0.79), husband (0.78) | 3/8 |
| window_size=5 | grateful (0.65), correspondence (0.64), entreating (0.63), residence (0.62), brother (0.61), solitary (0.61), tear (0.60), affectionate (0.60) | 1/8 |

### morning

| Run | Neighbors | Shared with baseline |
|---|---|---|
| baseline | evening (0.78), house (0.76), table (0.75), submitted (0.73), door (0.73), learning (0.73), affair (0.72), village (0.72) | 8/8 |
| D=100 | evening (0.79), house (0.76), table (0.75), door (0.74), submitted (0.74), inn (0.73), village (0.73), parsonage (0.72) | 6/8 |
| D=200 | evening (0.79), house (0.76), table (0.76), submitted (0.75), door (0.75), village (0.73), inn (0.73), ball (0.73) | 6/8 |
| learning_rate=0.005 | same (0.98), next (0.97), drawing (0.97), house (0.97), road (0.97), affair (0.97), room (0.97), subject (0.97) | 2/8 |
| learning_rate=0.1 | inn (0.60), protested (0.57), ensuring (0.54), observe (0.54), learning (0.53), morris (0.52), paces (0.50), mount (0.50) | 1/8 |
| epochs=1 | house (0.97), officers (0.97), conclusion (0.97), room (0.97), table (0.97), following (0.97), hall (0.97), housekeeper (0.96) | 2/8 |
| epochs=15 | subject (0.65), awoke (0.63), maunders (0.63), wrote (0.63), evening (0.62), time (0.61), prospect (0.60), sentence (0.60) | 1/8 |
| neg_samples=5 | evening (0.84), dining (0.81), house (0.81), parlour (0.80), ball (0.80), housekeeper (0.79), inn (0.79), breakfast (0.79) | 2/8 |
| window_size=5 | easter (0.73), next (0.71), evening (0.71), saturday (0.71), night (0.70), banished (0.70), ordered (0.69), day (0.69) | 1/8 |

### two

| Run | Neighbors | Shared with baseline |
|---|---|---|
| baseline | three (0.79), four (0.75), dances (0.71), older (0.71), motives (0.71), boasting (0.70), engagements (0.70), separating (0.68) | 8/8 |
| D=100 | three (0.76), four (0.73), dances (0.71), motives (0.69), older (0.68), boasting (0.68), regulate (0.68), engagements (0.68) | 7/8 |
| D=200 | three (0.78), four (0.74), dances (0.71), motives (0.70), boasting (0.70), older (0.69), regulate (0.68), engagements (0.68) | 7/8 |
| learning_rate=0.005 | parts (0.96), sense (0.96), living (0.96), gutenberg (0.96), expectation (0.96), expression (0.96), appearance (0.96), satisfaction (0.96) | 0/8 |
| learning_rate=0.1 | five (0.68), scarborough (0.65), four (0.64), older (0.59), bounds (0.59), three (0.59), french (0.59), lurking (0.58) | 3/8 |
| epochs=1 | three (0.97), importance (0.96), understanding (0.96), drawing (0.96), happiness (0.96), dining (0.96), matter (0.95), phrase (0.95) | 1/8 |
| epochs=15 | three (0.64), four (0.59), dances (0.56), older (0.55), eight (0.55), motives (0.55), french (0.55), suppressed (0.54) | 5/8 |
| neg_samples=5 | three (0.71), four (0.69), younger (0.62), proposed (0.61), older (0.61), descended (0.60), crowded (0.60), edged (0.60) | 3/8 |
| window_size=5 | three (0.72), gentlemen (0.70), weeks (0.69), eight (0.69), test (0.67), story (0.67), six (0.67), norris (0.67) | 1/8 |

### good

| Run | Neighbors | Shared with baseline |
|---|---|---|
| baseline | society (0.70), humoured (0.70), humour (0.69), kind (0.66), violent (0.66), small (0.66), pretty (0.65), wretched (0.65) | 8/8 |
| D=100 | humoured (0.67), humour (0.66), society (0.66), wretched (0.66), pretty (0.65), small (0.65), violent (0.65), accomplished (0.64) | 7/8 |
| D=200 | humoured (0.69), humour (0.68), society (0.68), wretched (0.67), violent (0.66), ill (0.66), small (0.66), wonderfully (0.65) | 6/8 |
| learning_rate=0.005 | half (0.94), fee (0.94), large (0.94), fine (0.94), woman (0.93), air (0.93), short (0.93), scheme (0.93) | 0/8 |
| learning_rate=0.1 | gayer (0.56), ill (0.54), vices (0.54), bright (0.51), harp (0.51), freshness (0.51), widow (0.51), large (0.50) | 0/8 |
| epochs=1 | cases (0.91), point (0.91), matter (0.91), most (0.91), piece (0.90), importance (0.90), relation (0.90), public (0.90) | 0/8 |
| epochs=15 | humoured (0.56), selfish (0.55), desirable (0.54), wonderful (0.52), vast (0.52), natural (0.52), unpretending (0.51), nay (0.50) | 1/8 |
| neg_samples=5 | humoured (0.74), ill (0.71), music (0.70), humour (0.70), kind (0.69), society (0.69), large (0.68), conceited (0.68) | 4/8 |
| window_size=5 | keeps (0.66), dinners (0.64), kindness (0.63), vouch (0.63), accomplished (0.62), ruin (0.62), blowzy (0.61), true (0.60) | 0/8 |
