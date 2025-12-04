![](https://img.shields.io/badge/Python-3.12-informational)
![](https://img.shields.io/badge/Status-Active-success)
![](https://img.shields.io/badge/License-MIT-inactive)

# RL Market Maker with Coevolving Traders

For methodology, research background, and implementation details, visit the companion [repository](https://github.com/Srithwak/Coevolutionary_ARL_vs_EGT_agents_research)

## Setup & Usage

### Clone and open the repository
```bash
git clone https://github.com/rangarajanpraneeth/coevolutionary-market-maker.git
code ./coevolutionary-market-maker
```

### Create and Activate Virtual Environment (Windows)
```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

### Create and Activate Virtual Environment (macOS/Linux)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Select the Interpreter in VS Code
Open **Command Palette** -> `Python: Select Interpreter`

Choose the one pointing to `.venv`

### Fetch OHLCV Data
|Flag (Required)|Description|
|-|-|
|`-s`, `--symbol`|Symbol -  `AAPL`, `NVDA`, `BTC`, `ETH`|
|Flag (Optional)|Default|Description|
|-|-|-|
|`-n`, `--name`|Lowercase symbol|Output filename (no `.csv`)|
|`-p`, `--period`|`6m`|Period - `1d`, `1w`, `1m`, `1y`, `max`|
|`-i`, `--interval`|`1d`|Interval - `1m`, `5m`, `15m`, `1h`, `1d`|
|`--start`|None|Explicit start date - `YYYY-MM-DD`|
|`--end`|None|Explicit end date - `YYYY-MM-DD`|
```bash
py ./src/helpers/fetchOHLCV.py -s NVDA -n nvda_up -p 5y -i 1d
py ./src/helpers/fetchOHLCV.py -s TSLA -n tsla_flat -p 5y -i 1d
py ./src/helpers/fetchOHLCV.py -s BTC -n btc_crash -p 1m -i 5m
```

### Plot OHLCV Data
|Flag (Required)|Description|
|-|-|
|`-f`, `--file`|Path to OHLCV CSV file|
```bash
py ./src/helpers/plotOHLCV.py -f ./data/NVDA/nvda_up.csv
py ./src/helpers/plotOHLCV.py -f ./data/TSLA/tsla_flat.csv
py ./src/helpers/plotOHLCV.py -f ./data/BTC/btc_crash.csv
```

### Run the Simulation
|Flag (Required)|Description|
|-|-|
|`-f`, `--file`|Path to OHLCV CSV file|
|Flag (Optional)|Default|Description|
|-|-|-|
|`-s`, `--seed`|`0`|Seed for reproducibility|
|`-o`, `--output`|Auto generated|Custom output directory|
```bash
py ./src/main.py -f ./data/NVDA/nvda_up.csv
py ./src/main.py -f ./data/TSLA/tsla_flat.csv
py ./src/main.py -f ./data/BTC/btc_crash.csv
```

## License
Copyright (c) 2025 Praneeth Rangarajan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**All rights reserved - license may change in the future.**
