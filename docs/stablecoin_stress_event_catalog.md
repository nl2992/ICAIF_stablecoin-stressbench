# Stablecoin Stress Event Catalog

Comprehensive historical catalogue of stablecoin stress events with data-tier classification
for use in the StressBench benchmark.

---

## Data Tier Classification

| Tier | Description | Computability | Benchmark use |
|------|-------------|--------------|---------------|
| **A** | Execution-grade: full L2 order book + VWAP labels computable | Full net_profit_bps labels | Primary benchmark tasks, oracle bound |
| **B** | Price-grade: OHLCV / DEX / trades available, but not full L2 | Price-basis labels only | Secondary analysis, illustrative depeg magnitudes |
| **C** | Context-grade: partial data or post-hoc reconstruction only | Historical context, qualitative | Literature cross-reference, stress taxonomy |

Tier determines which benchmark claims can be made about each event. Tier A events support
execution-aware claims (oracle_capture_pct, net_bps_captured). Tier B events support
price-dislocation claims only. Tier C events provide historical framing but no empirical claims.

---

## Event Entries

---

### Event 1: IRON/TITAN Collapse (June 2021)

**Event ID:** `iron_titan_2021`
**Tier:** C (context-grade)

**Mechanism:** Algorithmic stablecoin run. IRON was a partially-collateralized stablecoin backed
by USDC and TITAN (the governance token). A bank-run dynamic emerged: TITAN's price fell,
reducing IRON's collateralization, triggering redemptions, which sold TITAN, accelerating the
decline. IRON depegged to $0 within ~24 hours.

**Key dates:**
- 2021-06-16: TITAN begins sharp decline; IRON depeg starts
- 2021-06-17: TITAN price reaches ~$0; IRON collapses to $0

**Affected stablecoins:** IRON (to $0), no significant contagion to USDC/USDT

**Peak depeg magnitude:** IRON: 100% (terminal); USDC/USDT: negligible (<5 bps)

**Data availability:**
- CoinGecko/CMC price feeds: OHLCV available (Tier C because no traded exchange L2)
- Polygon DEX data: partial swaps data available via The Graph
- No CEX L2 depth available (IRON was DEX-only)

**Coverage score:** 0.25

**Why Tier C:**
- IRON was traded primarily on QuickSwap (Polygon DEX), not on Binance/Coinbase/Kraken
- No real L2 depth available for the primary StressBench venues
- Post-hoc reconstruction only; no live data captured

**Literature:** Klages-Mundt & Minca (2022) model the algorithmic stablecoin death spiral using
a bank-run framework. This event is the canonical example of the "undercollateralized algorithmic
stablecoin" failure mode.

**Empirical use:** Provides taxonomy context for algorithmic vs collateralized stablecoin risk.
Cannot be used in primary benchmark tasks.

**Notes:** IRON/TITAN established the death-spiral template later replicated by UST/LUNA at
much larger scale. The collapse happened in <24 hours with no circuit breaker, demonstrating
the speed risk of algorithmic pegs.

---

### Event 2: Terra/UST Collapse (May 2022)

**Event ID:** `terra_ust_2022`
**Tier:** B (price-grade)

**Mechanism:** Algorithmic stablecoin death spiral at scale. UST was backed by LUNA (Terra's
native token) through a burn/mint mechanism. Large UST withdrawals from Anchor Protocol
(~$14B TVL) triggered a bank-run. LUNA collapsed from ~$85 to <$0.01 in 7 days. UST
depegged from $1 to ~$0.02. Total market cap destruction: ~$40B.

**Key dates:**
- 2022-05-07: Large Curve pool UST withdrawals begin
- 2022-05-09: UST depeg breaks 0.90; LUNA begins sharp decline
- 2022-05-11: UST reaches $0.30; emergency LUNA mint begins
- 2022-05-13: LUNA below $0.01; UST at $0.08
- 2022-05-14: Terra chain halted

**Affected stablecoins:**
- UST: terminal depeg
- USDT: brief stress (Tether dropped to $0.9985 briefly), recovered same day
- USDC: minimal (<10 bps)
- DAI: minimal, though Curve 3pool rebalancing caused temporary liquidity pressure

**Peak depeg magnitude:** UST: -98%; USDT: -15 bps briefly; DAI: -8 bps briefly

**Data availability:**
- CEX OHLCV (Binance, Coinbase, Kraken): available via public APIs
- Binance spot order book snapshots: partial (not full L2 depth tape)
- Curve pool reserves: available via The Graph
- On-chain Terra transactions: available via Terra Explorer archives
- Full L2 depth tape for Binance/Coinbase at the time: not in benchmark dataset

**Coverage score:** 0.50

**Why Tier B:**
- CEX price and trade data available (sufficient for price-basis labels)
- No full L2 order book depth tape in benchmark dataset
- Contagion on USDC/USDT was too brief and small (<10 bps) for meaningful label construction
- UST was not traded on the primary benchmark venues (Binance listed USTC separately)

**Literature:** Briola et al. (2023) analyze the UST collapse using network analysis of Curve
liquidity pools. Kwon (2022) provides a post-mortem. The event is classified as `validation`
split in the benchmark's event_windows.yaml.

**Empirical use:** Validates that the benchmark's training data (which predates this event) does
not contain post-UST-collapse distribution. Illustrative of speed of algorithmic stablecoin failure.

**Notes:** Contagion to USDC/USDT was short-lived but real. Curve's 3pool UST imbalance was
a leading indicator ~12 hours before the full depeg (later documented by Cintra & Holloway 2023
using BOCPD on pool reserves).

---

### Event 3: FTX Collapse (November 2022)

**Event ID:** `ftx_collapse_2022`
**Tier:** B (price-grade)

**Mechanism:** Exchange credit and insolvency shock. FTX's balance sheet was revealed to be
insolvent (FTT tokens used as collateral for Alameda loans). CoinDesk report on Nov 2 triggered
a bank run. FTX halted withdrawals on Nov 8. CEO Sam Bankman-Fried arrested; FTX filed for
Chapter 11 on Nov 11. USDT briefly depegged due to exchange-specific premium/discount effects
on FTX vs external markets.

**Key dates:**
- 2022-11-02: CoinDesk publishes Alameda balance sheet
- 2022-11-06: Binance CEO tweets about FTT liquidation
- 2022-11-08: FTX halts withdrawals
- 2022-11-11: FTX Chapter 11 filing

**Affected stablecoins:**
- USDT: briefly traded at -5 to -20 bps on external CEX due to FTX-specific premium
- USDC: <5 bps impact
- DAI: <5 bps impact

**Peak depeg magnitude:** USDT: -20 bps on Kraken briefly; USDC: -5 bps

**Data availability:**
- CEX OHLCV (Binance, Coinbase, Kraken): available
- Binance partial order book data: available via Binance archive
- FTX internal data: not available (exchange defunct)
- On-chain USDT movements: available via Etherscan/Tether transparency

**Coverage score:** 0.50

**Why Tier B:**
- Primary venue data (Binance, Coinbase, Kraken) available at price level
- The stress was FTX-specific; the benchmark venues show only mild contagion
- Full L2 depth tape from Nov 2022 not in benchmark dataset
- Basis magnitudes were small compared to SVB event; label density too low for reliable
  training examples

**Literature:** Conlon, Corbet & McGee (2023) document the FTX contagion using event-study
methodology. Lyons & Viswanath-Natraj (2023) analyze the cross-venue price dynamics during
the FTX collapse.

**Empirical use:** Demonstrates exchange-specific credit shocks produce smaller stablecoin
dislocations than reserve-bank shocks (SVB). Useful for event taxonomy paper discussion.

**Notes:** The FTX collapse primarily affected crypto-specific risk (FTT, SOL) rather than
stablecoin pegs. This contrasts with the SVB event where the direct reserve-bank connection
created genuine USDC price risk.

---

### Event 4: BUSD Regulatory Winddown (February–March 2023)

**Event ID:** `busd_regulatory_2023`
**Tier:** B (price-grade)

**Mechanism:** Regulatory enforcement action. NYDFS ordered Paxos to stop minting new BUSD
on February 13, 2023. SEC issued a Wells notice to Paxos for BUSD being an unregistered
security. Binance converted user BUSD holdings to USDT/USDC. BUSD supply declined from $16B
to ~$8B within 30 days. Price impact was managed but brief dislocations occurred.

**Key dates:**
- 2023-02-13: NYDFS orders Paxos to stop BUSD minting
- 2023-02-13: SEC Wells notice to Paxos
- 2023-02-15: Binance announces BUSD conversion program
- 2023-03-01: BUSD supply below $10B (from $16B)

**Affected stablecoins:**
- BUSD: brief -10 to -30 bps dislocation during conversion rushes
- USDC: brief +5 to +10 bps appreciation as conversion destination
- USDT: brief +5 to +10 bps appreciation

**Peak depeg magnitude:** BUSD: -30 bps briefly; USDC/USDT: +10 bps appreciation

**Data availability:**
- CEX OHLCV for BUSD: available on Binance
- Partial order book data: available via Binance archive
- Paxos attestation reports: available (public)
- Full L2 depth tape: available via Binance archive for this period

**Coverage score:** 0.50

**Why Tier B:**
- BUSD is not a primary benchmark asset (benchmark focuses on USDC/USDT)
- The basis dislocations were smaller than the SVB event
- On-chain settlement data for BUSD conversion is available but not integrated

**Literature:** Gorton & Zhang (2023) analyze regulatory approaches to stablecoin issuers and
the systemic implications of abrupt winddown mandates.

**Empirical use:** Shows regulatory actions can create brief but predictable dislocations.
The slow unwinding (days, not minutes) creates different signal dynamics than the SVB event.

**Notes:** The BUSD winddown occurred 3 weeks before the SVB/USDC event. The two events together
may explain elevated stablecoin basis volatility in Q1 2023.

---

### Event 5: USDC/SVB Stress (March 10–15, 2023) — PRIMARY BENCHMARK EVENT

**Event ID:** `usdc_svb_2023`
**Tier:** A (execution-grade)

**Mechanism:** Reserve-bank insolvency shock. Silicon Valley Bank (SVB), where Circle held
approximately $3.3B of USDC reserves (~8% of reserves), was seized by FDIC regulators on
March 10, 2023. USDC depegged to as low as $0.87 on secondary markets. The depeg was driven by
uncertainty about reserve recovery. On March 12, the FDIC announced full deposit insurance for
SVB, resolving the uncertainty. USDC recovered to $0.997 by March 13 and to $1.000 by March 15.

**Key dates:**
- 2023-03-09: SVB announces emergency capital raise; deposits begin leaving
- 2023-03-10 08:30 UTC: FDIC seizure announcement
- 2023-03-10 10:00–18:00 UTC: USDC begins depegging; reaches $0.95 on Coinbase
- 2023-03-11: USDC at $0.87 on peak stress (Curve pool heavy USDC imbalance)
- 2023-03-12 22:00 UTC: US Treasury + Fed + FDIC joint statement: full SVB deposit insurance
- 2023-03-13: USDC rallies to $0.997
- 2023-03-15: USDC fully restored to $1.000

**Affected stablecoins:**
- USDC: peak -1300 bps (~13 cents off peg)
- DAI: secondary depeg (backed partly by USDC collateral); peak -200 bps
- USDT: brief premium +50 bps (flight to Tether)
- BUSD: already in winddown, minor incremental pressure

**Peak depeg magnitude:** USDC: -1300 bps; DAI: -200 bps; USDT: +50 bps (premium)

**Data availability:**
- Binance/Coinbase/Kraken real L2 snapshots: YES (captured during event)
- VWAP labels at $10K and $50K notional: YES (benchmark primary labels)
- Net profit bps labels (all notional sizes): YES
- Full 5-day window covered in test split: YES

**Coverage score:** 1.0

**Why Tier A:**
- Real L2 depth data captured during the event for all benchmark venues
- Full net_profit_bps labels computable at all notional sizes
- Primary benchmark test event; all paper claims reference this window
- Execution-grade: supports oracle_capture_pct, net_bps_captured, false-positive cost claims

**Literature:**
- Gorton & Zhang (2023) use this event to document stablecoin fragility under reserve uncertainty
- Catalini & de Gortari (2023) analyze the USDC depeg mechanism
- Hautsch, Scheuch & Voigt (2018) limits-to-arbitrage framework directly applicable to
  the execution-gap question

**Empirical use:** Primary benchmark test split. All paper claims anchored here. The 12× price-
to-execution gap (35% price-basis positive vs 2.88% executable) is measured during this window.
Oracle earns 161–225 net bps per trade on the test split.

**Notes:** The event lasted approximately 5 days from initial depeg to full recovery. The speed
of the recovery (once FDIC guarantee was announced) is consistent with reserves-based stablecoins
having a well-defined fair value anchor, unlike algorithmic stablecoins.

---

### Event 6: USDC Recovery Window (March 15–April 1, 2023)

**Event ID:** `usdc_svb_recovery_2023`
**Tier:** A/B (execution-grade for CEX, price-grade for on-chain recovery metrics)

**Mechanism:** Post-SVB recovery. After the FDIC announced full deposit insurance for SVB on
March 12, USDC reanchored to $1.000. The recovery window exhibits different dynamics from the
stress window: basis mean-reversion, decreasing volatility, reduced arbitrage opportunities.
Studying the recovery validates that models generalise across both stress and calm regimes.

**Key dates:**
- 2023-03-15: USDC at $1.000 restored
- 2023-03-20: Volatility returned to pre-SVB levels
- 2023-04-01: Window end (arbitrary recovery boundary)

**Affected stablecoins:** USDC: full recovery. DAI: full recovery. USDT: premium normalized.

**Peak depeg magnitude (during recovery):** USDC: <10 bps; routine cross-venue basis noise only

**Data availability:**
- L2 snapshots: available for CEX venues (Tier A for price prediction)
- Net profit labels: available but most windows are non-executable (label_arb = 0)
- On-chain metrics: partial (USDC reserve attestations quarterly only)

**Coverage score:** 0.75

**Why Tier A/B:**
- CEX L2 data available (Tier A for execution labels)
- But the economic content is low (very few profitable windows)
- On-chain recovery data (redemption volumes, reserve reconstitution) not fully integrated

**Empirical use:** Validates that model false-positive rate drops post-recovery, consistent with
regime change. Appears in event_windows.yaml as `usdc_depeg_2023_recovery` (split=test boundary).

**Notes:** The recovery window is economically important as a "normal" comparator within the
same data epoch. Models that overfit to the stress period should degrade here.

---

### Event 7: USDT/Curve Pool Stress (June 2023)

**Event ID:** `usdt_curve_2023`
**Tier:** B (price-grade)

**Mechanism:** Curve pool imbalance and brief USDT stress. In mid-June 2023, concerns about
Tether's commercial paper reserves re-emerged alongside a large Curve 3pool imbalance (USDT
became overweight in the pool as holders swapped out). USDT briefly traded at -8 bps on
Binance. The event was short-lived (hours) compared to the SVB event (days).

**Key dates:**
- 2023-06-12: Curve 3pool imbalance begins; USDT ~60% of pool (normally 33%)
- 2023-06-13: Peak USDT discount of -8 bps; rebalancing begins
- 2023-06-14: Pool rebalances; USDT returns to <2 bps discount
- 2023-06-15: Normal conditions restored

**Affected stablecoins:**
- USDT: peak -8 bps
- USDC: brief +3 bps premium (relative flight)
- DAI: +2 bps premium

**Peak depeg magnitude:** USDT: -8 bps; USDC: +3 bps

**Data availability:**
- CEX OHLCV (Binance/Coinbase/Kraken): available
- Curve pool reserves via The Graph: available
- On-chain USDT transfers: available via Etherscan/Nansen
- Full CEX L2 depth tape: partially available via Binance archive

**Coverage score:** 0.50

**Why Tier B:**
- Basis magnitudes were small (-8 bps vs -1300 bps for SVB)
- Event duration was hours, not days
- Curve on-chain data available but not integrated into benchmark dataset
- Insufficient profitable windows for meaningful label construction at $10K notional

**Literature:** Lyons & Viswanath-Natraj (2023) document Tether's reserve opacity as a
persistent source of brief periodic dislocations. Briola et al. (2023) analyze Curve pool
imbalances as leading indicators of stablecoin stress.

**Empirical use:** Illustrates that on-chain Curve pool data (not in benchmark) may provide
predictive signal for USDT stress. Motivates future data collection extension.

**Notes:** The June 2023 USDT event falls outside the benchmark test split. It is included
in the catalog to document the frequency of minor stablecoin stress events post-SVB.

---

## Coverage Summary

| Event | Tier | Coverage Score | Primary Use |
|-------|------|---------------|-------------|
| IRON/TITAN Jun 2021 | C | 0.25 | Taxonomy context |
| Terra/UST May 2022 | B | 0.50 | Validation split |
| FTX Collapse Nov 2022 | B | 0.50 | Illustrative context |
| BUSD Regulatory Feb–Mar 2023 | B | 0.50 | Illustrative context |
| USDC/SVB Mar 10–15 2023 | **A** | **1.00** | **Primary benchmark test** |
| USDC Recovery Mar 15–Apr 1 2023 | A/B | 0.75 | Test split recovery window |
| USDT/Curve Jun 2023 | B | 0.50 | Out-of-sample context |

---

## Implications for Benchmark Claims

Claims about execution-grade arbitrage (net_bps_captured, oracle_capture_pct, execution gap)
are valid ONLY for Tier A events (USDC/SVB primary, USDC recovery).

Claims about price-basis dynamics (basis magnitude, depeg frequency) can reference Tier B events
for illustrative purposes but MUST be clearly labeled as "price-grade" claims.

Claims about historical taxonomy (which stablecoin failure modes exist, what mechanisms apply)
can reference Tier C events with appropriate caveats about data availability.

---

## References

- Briola, A., et al. (2023). Anatomy of a run: The Terra Luna crash. *Finance Research Letters*.
- Catalini, C., & de Gortari, A. (2023). On the economic design of stablecoins. *MIT DCI Working Paper*.
- Cintra, R., & Holloway, T. (2023). Bayesian changepoint detection in Curve stablecoin pools.
- Conlon, T., Corbet, S., & McGee, R. (2023). The FTX collapse and systemic crypto risk.
- Gorton, G., & Zhang, J. (2023). Taming wildcat stablecoins. *University of Chicago Law Review*.
- Hautsch, N., Scheuch, C., & Voigt, S. (2018). Limits to arbitrage in markets with stochastic
  settlement latency. *Journal of Financial Markets*.
- Klages-Mundt, A., & Minca, A. (2022). While stability lasts: A stochastic model of
  stablecoin pegs. *Operations Research*.
- Lyons, R., & Viswanath-Natraj, G. (2023). What keeps stablecoins stable?
  *Journal of International Money and Finance*.
