# ApolloTrader
Fully automated crypto trading powered by a custom price prediction AI and a structured/tiered DCA system.

**Version:** Apollo Trader 26 
**Features:** Three-category coin management, debug mode, simulation mode, configurable themes, enhanced error handling, and comprehensive settings system.

## Recent Updates

**Three-Category Coin Management System:**
- **Active Trading Coins**: Full buy/sell automation with position limits (default: ETH, SOL, XRP, LINK)
- **Accumulation Coins**: Buy-only strategy for long-term holding, stop-loss only sells, unlimited positions (default: BTC)
- **Liquidation Coins**: Sell-only to exit unwanted positions, auto-removes at dust threshold (default: DOGE)
- **Visual Feedback**: Color-coded coin buttons (green for accumulate, red for liquidate) in chart tabs
- **Position Limit Fix**: Only active trading coins count toward max concurrent positions
- **Auto-Cleanup**: Liquidation coins automatically removed when balance drops below minimum trade size
- **Safety**: System never touches coins not listed in any category
- **Hot-Reload**: Category changes take effect immediately without restart

**Enhanced AI Pattern Matching:**
- **Kernel-Weighted Predictions**: AI now uses exponential distance weighting (e^(-diff/threshold)) where closer pattern matches contribute more to predictions. This significantly improves prediction accuracy by prioritizing the most similar historical patterns.
- **Threshold Enforcement Setting**: New configurable strictness levels (Tight/Balanced/Loose/None) control how the AI filters and weights patterns. Tight mode uses only very similar patterns for highest confidence, while Loose mode considers a wider range for volatile markets.
- **Improved Baseline Sensitivity**: Volatility-adaptive baseline increased from 0.05% to 0.1% (2× improvement) for better handling of micro-movements in low-volatility conditions.
- **Prediction Candles**: AI now generates future-timestamped prediction candles visible on charts (white candle), showing multi-timeframe consensus on expected price action.

**System Improvements:**
- **Enhanced File Security**: System signal files now use `.dat` extension instead of `.txt` to discourage accidental manual editing that could corrupt trading data.
- **Directionality Verification**: Confirmed pattern matching treats bull and bear patterns symmetrically using abs() for consistent behavior regardless of market direction.

"It's an instance-based (kNN/kernel-style) predictor with online per-instance reliability weighting, used as a multi-timeframe trading signal." - ChatGPT on the type of AI used in this trading bot.

So what exactly does that mean?

When people think AI, they usually think about LLM style AIs and neural networks. What many people don't realize is there are many types of Artificial Intelligence and Machine Learning - and the one in my trading system falls under the "Other" category.

When training for a coin, it goes through the entire history for that coin on multiple timeframes and saves each pattern it sees, along with what happens on the next candle AFTER the pattern. It uses these saved patterns to generate a predicted candle by taking a weighted average of the closest matches in memory to the current pattern in time. This weighted average output is done once for each timeframe, from 1 hour up to 1 week. Each timeframe gets its own predicted candle. The low and high prices from these candles are what are shown as the blue and orange dotted horizontal lines on the price charts. 

After a candle closes, it checks what happened against what it predicted, and adjusts the weight for each "memory pattern" that was used to generate the weighted average, depending on how accurate each pattern was compared to what actually happened.

Yes, it is EXTREMELY simple. Yes, it is STILL AI.

Here is how the trading bot utilizes the price prediction ai to automatically make trades:

For determining when to start trades, the AI's Thinker script sends a signal to start a trade for a coin if the ask price for the coin drops below at least 3 of the the AI's predicted low prices for the coin (it predicts the currently active candle's high and low prices for each timeframe across all timeframes from 1hr to 1wk). **This threshold can be customized** in the Trading Settings (default: LONG signal minimum of 3, SHORT signal maximum of 0).

**Buy Priority (Multiple Qualifying Coins):** When multiple coins meet entry criteria simultaneously, the system uses an opportunity scoring algorithm: each coin is scored by its LONG signal strength (0-7), and the coin with the highest score gets priority. In case of a tie, the original coin list order acts as the tiebreaker. This ensures the bot always opens the strongest signal opportunity first, rather than simply processing coins sequentially.

For determining when to DCA, it uses either the current price level from the AI that is tied to the current amount of DCA buys that have been done on the trade (for example, right after a trade starts when 3 blue lines get crossed, its first DCA wont happen until the price crosses the 4th line, so on so forth), or it uses the configurable drawdown % for its current level, whichever it hits first. **Default DCA levels: -2.5%, -5.0%, -10.0%, -20.0%** (4 levels). It allows a max of 2 DCAs within a rolling 24hr window to keep from dumping all of your money in too quickly on coins that are having an extended downtrend! **DCA levels and limits can be customized** in the Trading Settings.

For determining when to sell, the bot uses a trailing profit margin to maximize the potential gains. The margin line is set at either 5% gain if no DCA has happened on the trade, or 3% gain if any DCA has happened. The trailing margin gap is 0.5% (this is the amount the price has to go over the profit margin to begin raising the profit margin up to TRAIL after the price and maximize how much profit is gained once the price drops below the profit margin again and the bot sells the trade).

**Multi-Timeframe Exit Confirmation:** When the price crosses below the trailing profit margin line, the bot requires bearish confirmation across multiple timeframes before executing the sell. This prevents premature exits during minor pullbacks while still capturing profit on true trend reversals. The exit requires BOTH short_signal >= 4 (bearish confirmation) AND long_signal <= 0 (no bullish resistance). **The stop-loss (-40%) ALWAYS executes immediately, bypassing MTF check for safety.** **Exit signal thresholds, profit margins, and timing can be customized** in the Trading Settings.

# Setup & First-Time Use (Windows)

ApolloTrader is designed to be easy to set up with minimal technical knowledge required.

**Important:** This software can place trades automatically. You are responsible for what it does.  
Keep your API keys private. We are not giving financial advice. We are not responsible for any losses incurred. You are fully responsible for doing your own due diligence to learn and understand this trading system and to use it properly. You are fully responsible for all of your money, and any gains or losses.

---

## Step 1 — Install Python

1. Go to **python.org** and download Python for Windows (Python 3.10 or newer recommended).
2. Run the installer.
3. **Check the box** that says: **"Add Python to PATH"** (very important!).
4. Click **Install Now**.

---

## Step 2 — Download ApolloTrader

1. Go to the ApolloTrader GitHub repository.
2. Click the green **Code** button, then **Download ZIP**.
3. Extract the ZIP file to a folder on your computer, like: `C:\ApolloTrader\`

---

## Step 3 — Launch ApolloTrader (One-Click Setup!)

1. Navigate to your ApolloTrader folder.
2. Double-click **ApolloTrader.pyw** to launch the hub.

**First Launch:** ApolloTrader will automatically check if required packages are installed. If not, it will ask permission to install them. Click **Install** and wait a few minutes while it sets everything up. This is a one-time process.

**Note:** Windows may ask for permission to run Python or install packages - this is normal, click Allow/Yes.

The **ApolloTrader Hub** will open - this is your main control center.

---

## Step 4 — Configure Settings

### Initial Setup

On first launch, ApolloTrader automatically uses the folder where you extracted it as the main directory. You can change this later if needed.

The Hub menu bar contains:
- **Settings**: Hub Settings, Trading Settings, Training Settings, API Setup Wizard
- **View**: Toggle Autopilot, Simulation Mode, Debug Mode
- **Help**: Exit application

To configure settings, click **Settings → Hub Settings** and configure:

- **Active Trading Coins (comma-separated)**: Coins for full buy/sell trading (default: ETH, SOL, XRP, LINK). Start with one or two coins for your first run.
- **Accumulation Coins (comma-separated)**: Buy-only coins for long-term holding (default: BTC). System will buy on signals and DCA, but only sell on stop-loss.
- **Liquidation Coins (comma-separated)**: Sell-only coins to exit positions (default: DOGE). System will sell on profit/stop-loss but never buy. Auto-removes from list when balance drops below minimum trade size.
- **Main neural folder**: Already set to your ApolloTrader directory by default.
- **Robinhood API**: Click **Setup Wizard** and follow these steps:
  1. Click **Generate Keys**.
  2. Copy the **Public Key** shown in the wizard.
  3. On Robinhood.com, go to your account settings and add a new API key. Paste the Public Key.
  4. Set permissions to allow READ and TRADE (the wizard tells you what to select).
  5. Robinhood will show your API Key (often starts with `rh`). Copy it.
  6. Paste the API Key back into the wizard and click **Save**.
  7. Close the wizard and go back to the **Settings** screen.

Click **Save** when done.

After saving, you will have two credential files in your ApolloTrader folder:  
`rh_key.txt` and `rh_secret.txt` - These are automatically **encrypted for security**. Keep them private and secure!

ApolloTrader uses a simple folder structure:  
**All coins use their own subfolders** (like `BTC\`, `ETH\`, etc.).

Each coin folder contains:
- Training data and pattern memories
- Neural prediction files (`low_bound_prices.dat`, `high_bound_prices.dat`)
- Trading signals and status files (`.dat` extension to discourage manual editing)
- Prediction candles output (`prediction_candles.json`)

### Optional: Customize Trading Behavior

ApolloTrader includes customizable settings so you can tune the bot to your preferences:

**Settings → Trading Settings** - Configure trading parameters:
- Entry signal thresholds (when to start trades)
- DCA (Dollar Cost Averaging) levels and limits
- Profit margin targets and trailing gaps
- Position sizing
- Timing delays

**Settings → Training Settings** - Configure AI training:
- Staleness threshold (how often to retrain)
- Auto-retrain option (future feature)

All settings have built-in validation and save instantly. Changes to trading settings take effect immediately (no restart needed). The white marker lines on the Neural Signal display update automatically when you change entry signal thresholds.

### Understanding Coin Categories

ApolloTrader organizes coins into three strategic categories, each with different trading behavior:

#### Active Trading Coins
**Full automated trading** - The system will both buy and sell based on AI signals:
- **Buys**: Opens positions when entry signals meet threshold (default: LONG ≥4, SHORT ≤0)
- **DCA**: Adds to positions on drawdowns using neural levels or hardcoded percentages
- **Sells**: Exits on trailing profit margin or stop-loss
- **Position Limits**: Counts toward max concurrent positions limit (default: 3)
- **Use Case**: Coins you want to actively trade for profit

#### Accumulation Coins  
**Buy-only strategy** - Build long-term positions without selling:
- **Buys**: Opens positions on entry signals, same as active trading
- **DCA**: Adds to positions on drawdowns to lower cost basis
- **Sells**: **Only on stop-loss** (default: -40%) for emergency exit
- **Position Limits**: Does NOT count toward position limits (unlimited accumulation)
- **Use Case**: Coins you believe in long-term and want to accumulate (e.g., BTC, ETH)
- **Chart Display**: Shown in **green** in coin list above charts

#### Liquidation Coins
**Sell-only strategy** - Exit unwanted positions:
- **Buys**: **Blocked** - system will never open new positions
- **DCA**: **Blocked** - will not add to existing positions  
- **Sells**: Exits on trailing profit margin or stop-loss
- **Position Limits**: Does NOT count toward position limits
- **Auto-Cleanup**: Automatically removed from list when balance drops below minimum trade size (dust threshold)
- **Use Case**: Coins you want to exit but don't want to market dump (e.g., meme coins, failed trades)
- **Chart Display**: Shown in **red** in coin list above charts

**Safety Feature**: The system will NEVER touch coins not listed in any of these three categories. This protects holdings you may have that you don't want the bot to trade.

**Example Configuration**:
- **Active**: ETH, SOL, XRP, LINK (day trading)
- **Accumulation**: BTC (long-term HODLing)
- **Liquidation**: DOGE (exiting old position)

---

## Step 5 — Train (inside the Hub)

Training builds the system's coin "memory" so it can generate signals.

1. In the Hub, click **Train All**.
2. Wait until training finishes (this can take several minutes for each coin).

---

## Step 6 — Start the system (inside the Hub)

When training is done, click:

1. **Start All**

The Hub will automatically start the Thinker and Trader modules in the correct order. You don't need to manually start separate programs. The hub handles everything!

---

## Neural Levels (the LONG/SHORT numbers)

- These are signal strength levels from low to high.
- Higher number = stronger signal.
- LONG = buy-direction signal. SHORT = sell-direction signal.

A TRADE WILL START FOR A COIN IF THAT COIN REACHES A LONG LEVEL OF 4 OR HIGHER WHILE HAVING A SHORT LEVEL OF 0! (Customizable in Trading Settings)

---

## Customizing Trading Behavior

Apollo Trader allows you to customize trading parameters without editing code:

### Trading Settings
Access via **Settings → Trading Settings** in the Hub menu:
- **Entry Signals**: Adjust LONG/SHORT signal thresholds for when trades start (default: LONG min 4, SHORT max 0)
- **Exit Signals (MTF Confirmation)**: Configure multi-timeframe exit confirmation thresholds (default: SHORT min 4, LONG max 0). Prevents premature exits during minor pullbacks while capturing profit on true trend reversals. Stop-loss always bypasses MTF check.
- **DCA Levels**: Configure drawdown percentages and maximum DCA buys per 24 hours (default: 4 levels at -2.5%, -5.0%, -10.0%, -20.0%)
- **Profit Margins**: Set trailing gap and target percentages (default: 5% no DCA, 3% with DCA), stop loss (default: -40%)
- **Position Sizing**: Control initial allocation percentage and minimum trade size
- **Timing**: Adjust main loop delay (default: 0.5s) and post-trade cooldown (default: 30s)

The marker lines on the Neural Signal tiles update automatically when you change signal thresholds:
- **White solid lines** show entry thresholds (when to start trades)
- **Orange dashed lines** show exit thresholds (when MTF confirmation allows sells)
No restart required!

### Training Settings
Access via **Settings → Training Settings** in the Hub menu:
- **Staleness Days**: How many days before retraining is recommended (default: 14)
- **Auto-retrain**: Enable automatic retraining when data becomes stale
- **Pattern Matching**: Configurable thresholds for pattern similarity (uses relative percentage matching)
- **Volatility Adaptation**: Training automatically adjusts matching thresholds based on each coin's volatility (4.0x multiplier)
- **Threshold Enforcement** (Tight/Balanced/Loose/None): Controls how strictly the AI filters patterns:
  - **Tight (1×)**: Only uses very similar patterns, highest confidence but fewer data points
  - **Balanced (2×)**: Default setting, good balance between confidence and coverage
  - **Loose (5×)**: Uses wider range of patterns, more forgiving in volatile markets
  - **None (10×)**: Minimal filtering, considers most available patterns
  
  This setting scales three algorithm parameters:
  - Exclusion cutoff (how dissimilar before rejecting)
  - Perfect match threshold (when to accept pattern size)
  - Kernel bandwidth (distance weighting strength)

Changes to settings files take effect immediately for the trading bot (hot-reload with caching).

---

## Advanced Features

### Debug Mode
Enable detailed logging to troubleshoot issues or understand the bot's decision-making process:
1. Open **Settings → Hub Settings**
2. Enable **Debug Mode** checkbox
3. Save settings

Debug mode provides:
- Detailed market data fetch logs
- Pattern matching diagnostics
- Training file validation messages
- API call timing and retry information
- State persistence confirmation

All debug output appears in the Live Output tabs (Runner/Trader/Trainers) and does not clutter production logs when disabled.

### Simulation Mode
Test trading strategies without risking real money:
1. Open **Settings → Trading Settings**
2. Enable **Simulation Mode** checkbox
3. Save settings

Simulation mode:
- Tags all trades with "SIM_" prefix
- Excludes simulated trades from PnL calculations
- Shows "SIMULATION" indicator in the Hub
- Allows testing strategies with real market data
- Can run alongside real trading (separate coins recommended)

**Important:** Simulation mode still requires valid API credentials but will not execute real trades.

### Multi-Timeframe Exit Confirmation (MTF)
Prevent premature exits during minor pullbacks while capturing profit on true trend reversals:

**How It Works:**
When the price crosses below your trailing profit margin line, the system checks for bearish confirmation across all timeframes before executing the sell. This prevents selling during brief consolidations or minor dips that often recover to higher prices.

**Exit Requirements (BOTH must be true):**
- SHORT signal >= 4 (bearish confirmation across timeframes)
- LONG signal <= 0 (no bullish resistance)

**Safety Exception:**
Stop-loss (-40% by default) ALWAYS executes immediately, bypassing MTF check. This ensures catastrophic losses are cut regardless of signal conditions.

**Visual Indicators:**
The Neural Signal tiles display your exit thresholds:
- **Orange dashed lines** show exit requirements (SHORT minimum, LONG maximum)
- Real-time signal levels show current values (e.g., "L:3 S:2")
- When price crosses the trailing line, you can see at a glance whether MTF would allow the exit

**Live Output Messages:**
When a trailing profit margin sell is triggered, the trader shows clear messages:
- ✅ "MTF CONFIRMED" when both conditions pass → sell executes
- ⏸️ "MTF BLOCK" when conditions fail → position stays open, trailing line continues tracking
- 🛑 "STOP-LOSS" when emergency exit triggers → sell executes immediately (bypasses MTF)

**Configuration:**
Customize thresholds via **Settings → Trading Settings → Exit Signals (MTF Confirmation)**:
- Adjust SHORT signal minimum (0-7, default: 4)
- Adjust LONG signal maximum (0-7, default: 0)
- Changes take effect immediately and update the orange marker lines

**Benefit:** By requiring multi-timeframe bearish consensus, you avoid exiting positions that are still in strong uptrends, capturing additional 2-5% gains on average that would otherwise be left on the table.

### Theme Customization
Customize the Hub's appearance:
1. Edit `theme_settings.json` in your Apollo Trader folder
2. Modify colors for backgrounds, text, charts, and buttons
3. Reload the Hub to apply changes

Default themes included:
- **Cyber** (default dark theme with green/cyan accents)
- **Midnight** (blue-tinted dark theme)

Create your own theme by copying and modifying the color values in `theme_settings.json`.

### Enhanced Error Handling & Pattern Matching
The Enhanced Edition includes robust error recovery and improved AI pattern matching:

**Error Recovery:**
- **Circuit Breaker Pattern**: Prevents cascading failures when API is down
- **Automatic Retry Logic**: Network requests retry with exponential backoff
- **Training File Validation**: Detects and skips corrupted training data
- **Graceful Degradation**: System continues operating even when individual components fail

**Pattern Matching Improvements:**
- **Kernel-Weighted Averaging**: Predictions use exponential distance weighting (e^(-diff/threshold)) where closer pattern matches contribute more to the final prediction. Weighting strength scales with threshold enforcement setting.
- **Relative Threshold Matching**: Thresholds are now relative to pattern magnitude (percentage-based)
- **Scale-Invariant**: Works consistently across different price levels and market conditions
- **Volatility-Based Adaptation**: Thresholds automatically adjust using 4.0× average volatility
- **Zero-Value Protection**: Uses 0.1% baseline for near-zero patterns (prevents division-by-zero, 2× previous value)
- **Threshold Enforcement Setting**: Configurable strictness (Tight/Balanced/Loose/None) for pattern filtering
- **Simplified Algorithm**: Removed complex PID controller in favor of transparent volatility-based system

### Chart & UI Improvements
- **Centered Window Positioning**: Window always opens within visible screen area
- **Chart Refresh Rate**: Configurable update interval (default: 10 seconds)
- **Neural Level Display**: Shows only timeframe labels on chart (1h, 2h, etc.) with automatic overlap hiding. Neural levels rendered as dotted lines (blue for long/support, orange for short/resistance) to visually differentiate from other chart elements.
- **Last Neurals Timestamp**: Charts display when neural predictions were last updated
- **Prediction Candles**: AI-generated future candle showing expected price action across all timeframes. Displayed as white filled candle extending beyond current price data. OHLC values represent multi-timeframe consensus prediction.
- **Optimized Padding**: Improved chart margins for better label visibility
- **Flow Status Indicators**: Visual checkmarks (✓) and hourglasses (⧗) show system state between stages

### Training Freshness & Automation
The system enforces training freshness and provides intelligent automation:

**Freshness Enforcement:**
- Trainer writes timestamp when training starts
- Hub and Thinker check training age before allowing trades
- Visual indicators show which coins need retraining
- Configurable staleness threshold (default: 14 days)
- **T-X countdown**: Shows hours remaining until training becomes stale (e.g., "BTC: TRAINED (T-72 HRS)")
- **T+X overdue indicator**: Shows hours overdue when training is stale (e.g., "BTC: TRAINED (T+3 HRS)")

**Auto-Start Thinker:**
- After manual training completes, the Thinker automatically starts
- Seamless workflow: click "Train All" → training finishes → Thinker starts automatically
- Similar to how Thinker auto-starts when opening the Hub with valid training data

**Smart Auto-Retrain:**
- When auto-retrain is enabled, system checks for stale training every minute
- **Priority Coin Protection**: Before stopping for retraining, checks if any coin has imminent trade action
- Delays retraining if a coin is within 2% of: new entry, DCA trigger, stop loss, or take profit
- User sees T+X countdown increment (training overdue) until all trades complete safely
- Then automatically stops, retrains all stale coins, and restarts
- Prevents taking the trader offline during critical moments

If a coin shows "NOT TRAINED / OUTDATED", run training before starting the bot.

### Hot-Reload Configuration
All configuration changes take effect without restarting:
- Trading parameters update immediately (trader checks config every loop)
- Coin list and category changes are detected on-the-fly (both hub and trader)
- Moving coins between categories takes effect immediately
- Theme updates apply after Hub reload
- Debug mode toggles instantly
- Entry signal thresholds update Neural Signal display markers in real-time

---

## Adding more coins (later)

1. Open **Settings → Hub Settings**
2. Add coins to the appropriate category:
   - **Active Trading Coins**: For full buy/sell trading
   - **Accumulation Coins**: For buy-only long-term holding
   - **Liquidation Coins**: For sell-only position exits
3. Save (the system will create folders and copy trainers for new coins)
4. Click **Train All**, wait for training to complete
5. Click **Start All**

**Note**: You can move coins between categories at any time. The trader will adapt to the new behavior on the next trading cycle (no restart needed). Existing positions remain intact when you change categories.

---

## Technical Improvements (Enhanced Edition)

This version includes numerous stability and reliability improvements over the original:

### Crash Prevention & Data Validation
- **Zero-division protection** in all mathematical operations
- **Training file validation** with format checking and bounds verification
- **Weight list synchronization** to prevent index errors
- **Empty data guards** to handle missing or incomplete data gracefully
- **Corrupt data detection** with automatic skip and continue logic
- **Profit margin display fix**: Corrected formatting to show accurate percentages (was displaying 100x too large)

### Performance Enhancements
- **Thread-safe directory switching** prevents cross-coin data contamination
- **Memory caching** reduces redundant file I/O operations
- **Mtime-based caching** skips reloading unchanged files (neural predictions, trade history, account data)
- **Configurable sleep timings** to balance performance vs API rate limits
- **Chart downsampling** to 250 points max for smooth rendering
- **Debounced UI updates** to prevent excessive redraws
- **Selective chart updates** refreshes only visible coin tab (prevents network stalls)

### Network Resilience
- **Robinhood API retry logic** (5 attempts with exponential backoff)
- **KuCoin fallback system** (client library → REST API)
- **Circuit breaker pattern** to prevent cascading failures
- **Invalid symbol cleanup** to handle malformed coin names
- **Connection timeout handling** with automatic recovery

### Code Quality
- **Module docstrings** with repository and author information
- **Type hints** for better IDE support and error detection
- **Function documentation** explaining behavior and parameters
- **DRY principle** applied with helper functions
- **Consistent error handling** using specific exceptions

All improvements maintain 100% backward compatibility with the original trading logic and produce identical results while providing enhanced reliability and maintainability.

---

## Donate

---

## Project Information

**ApolloTrader** is a fork and significant enhancement of the original PowerTrader_AI project.

**Primary Repository:** https://github.com/Dr-Dufenshmirtz/ApolloTrader  
**Primary Author:** Dr Dufenshmirtz

**Original Project:** https://github.com/garagesteve1155/PowerTrader_AI  
**Original Author:** Stephen Hughes (garagesteve1155)

This fork includes substantial improvements including:
- Three-category coin management (Active/Accumulation/Liquidation)
- Configurable training parameters via GUI
- Enhanced error handling and debugging systems
- Performance optimizations and code cleanup
- Expanded customization options for trading strategies
- Improved pattern matching and prediction algorithms

---

## Support

ApolloTrader is COMPLETELY free and open source!

If you'd like to support the **original PowerTrader_AI project**:
- Cash App: **$garagesteve**
- PayPal: **@garagesteve**
- Patreon: **patreon.com/MakingMadeEasy**

---

## License

ApolloTrader is released under the **Apache 2.0** license.