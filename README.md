# ApolloTrader
Fully automated crypto trading powered by a custom price prediction AI and a structured/tiered DCA system.

**Version:** Apollo 26 
**Features:** Debug mode, simulation mode, configurable themes, enhanced error handling, and comprehensive settings system.

"It's an instance-based (kNN/kernel-style) predictor with online per-instance reliability weighting, used as a multi-timeframe trading signal." - ChatGPT on the type of AI used in this trading bot.

So what exactly does that mean?

When people think AI, they usually think about LLM style AIs and neural networks. What many people don't realize is there are many types of Artificial Intelligence and Machine Learning - and the one in my trading system falls under the "Other" category.

When training for a coin, it goes through the entire history for that coin on multiple timeframes and saves each pattern it sees, along with what happens on the next candle AFTER the pattern. It uses these saved patterns to generate a predicted candle by taking a weighted average of the closest matches in memory to the current pattern in time. This weighted average output is done once for each timeframe, from 1 hour up to 1 week. Each timeframe gets its own predicted candle. The low and high prices from these candles are what are shown as the blue and orange horizontal lines on the price charts. 

After a candle closes, it checks what happened against what it predicted, and adjusts the weight for each "memory pattern" that was used to generate the weighted average, depending on how accurate each pattern was compared to what actually happened.

Yes, it is EXTREMELY simple. Yes, it is STILL AI.

Here is how the trading bot utilizes the price prediction ai to automatically make trades:

For determining when to start trades, the AI's Thinker script sends a signal to start a trade for a coin if the ask price for the coin drops below at least 3 of the the AI's predicted low prices for the coin (it predicts the currently active candle's high and low prices for each timeframe across all timeframes from 1hr to 1wk). **This threshold can be customized** in the Trading Settings (default: LONG signal minimum of 3, SHORT signal maximum of 0).

For determining when to DCA, it uses either the current price level from the AI that is tied to the current amount of DCA buys that have been done on the trade (for example, right after a trade starts when 3 blue lines get crossed, its first DCA wont happen until the price crosses the 4th line, so on so forth), or it uses the configurable drawdown % for its current level, whichever it hits first. It allows a max of 2 DCAs within a rolling 24hr window to keep from dumping all of your money in too quickly on coins that are having an extended downtrend! **DCA levels and limits can be customized** in the Trading Settings.

For determining when to sell, the bot uses a trailing profit margin to maximize the potential gains. The margin line is set at either 5% gain if no DCA has happened on the trade, or 2.5% gain if any DCA has happened. The trailing margin gap is 0.5% (this is the amount the price has to go over the profit margin to begin raising the profit margin up to TRAIL after the price and maximize how much profit is gained once the price drops below the profit margin again and the bot sells the trade). **All profit margin percentages can be customized** in the Trading Settings.

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

In the Hub, click **Settings → Hub Settings** and configure:

- **Coins (comma-separated)**: Start with **BTC** for your first run. Add more coins later (ETH, XRP, BNB, DOGE, etc.).
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

A TRADE WILL START FOR A COIN IF THAT COIN REACHES A LONG LEVEL OF 3 OR HIGHER WHILE HAVING A SHORT LEVEL OF 0! (Customizable in Trading Settings)

---

## Customizing Trading Behavior

Apollo Trader allows you to customize trading parameters without editing code:

### Trading Settings
Access via **Settings → Trading Settings** in the Hub menu:
- **Entry Signals**: Adjust LONG/SHORT signal thresholds for when trades start
- **DCA Levels**: Configure drawdown percentages and maximum DCA buys per 24 hours
- **Profit Margins**: Set trailing gap and target percentages (with/without DCA)
- **Position Sizing**: Control initial allocation percentage and minimum trade size
- **Timing**: Adjust main loop delay and post-trade delays

The white marker lines on the Neural Signal tiles update automatically when you change entry signal settings - no restart required!

### Training Settings
Access via **Settings → Training Settings** in the Hub menu:
- **Staleness Days**: How many days before retraining is recommended (default: 14)
- **Auto-retrain**: Enable automatic retraining when data becomes stale (future feature)

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

### Theme Customization
Customize the Hub's appearance:
1. Edit `theme_settings.json` in your Apollo Trader folder
2. Modify colors for backgrounds, text, charts, and buttons
3. Reload the Hub to apply changes

Default themes included:
- **Cyber** (default dark theme with green/cyan accents)
- **Midnight** (blue-tinted dark theme)

Create your own theme by copying and modifying the color values in `theme_settings.json`.

### Enhanced Error Handling
The Enhanced Edition includes robust error recovery:
- **Circuit Breaker Pattern**: Prevents cascading failures when API is down
- **Automatic Retry Logic**: Network requests retry with exponential backoff
- **Training File Validation**: Detects and skips corrupted training data
- **Graceful Degradation**: System continues operating even when individual components fail

### Training Freshness Gates
The system now enforces training freshness to prevent trading with outdated predictions:
- Trainer writes timestamp when training starts
- Hub and Runner check training age before allowing trades
- Visual indicators show which coins need retraining
- Configurable staleness threshold (default: 14 days)

If a coin shows "NOT TRAINED / OUTDATED", run training before starting the bot.

### Hot-Reload Configuration
All configuration changes take effect without restarting:
- Trading parameters update immediately
- Coin list changes are detected on-the-fly
- Theme updates apply after Hub reload
- Debug mode toggles instantly

---

## Adding more coins (later)

1. Open **Settings → Hub Settings**
2. Add one new coin
3. Save
4. Click **Train All**, wait for training to complete
5. Click **Start All**

---

## Technical Improvements (Enhanced Edition)

This version includes numerous stability and reliability improvements over the original:

### Crash Prevention & Data Validation
- **Zero-division protection** in all mathematical operations
- **Training file validation** with format checking and bounds verification
- **Weight list synchronization** to prevent index errors
- **Empty data guards** to handle missing or incomplete data gracefully
- **Corrupt data detection** with automatic skip and continue logic

### Performance Enhancements
- **Thread-safe directory switching** prevents cross-coin data contamination
- **Memory caching** reduces redundant file I/O operations
- **Configurable sleep timings** to balance performance vs API rate limits
- **Chart downsampling** to 250 points max for smooth rendering
- **Debounced UI updates** to prevent excessive redraws

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

Apollo Trader is COMPLETELY free and open source! If you want to support the project:

- Cash App: **$garagesteve**
- PayPal: **@garagesteve**
- Patreon: **patreon.com/MakingMadeEasy**

---

## Contributors

- **Stephen Hughes (garagesteve1155)** - Original author and maintainer
- **Dr-Dufenshmirtz** - Enhanced Edition improvements

---

## License

Apollo Trader is released under the **Apache 2.0** license.