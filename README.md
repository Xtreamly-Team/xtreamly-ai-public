# 🌐 Xtreamly Volatility Public

Welcome to the **Xtreamly Volatility Public** repository! This project supports Xtreamly's AI-driven research focused on modeling and forecasting price volatility in cryptocurrency markets, especially for **Ethereum (ETH)** and **Bitcoin (BTC)**.

Here you’ll find open-source implementations of state-of-the-art volatility models, historical datasets, and documentation to help you replicate, evaluate, or extend our work.

---

## 📖 Overview

This repository provides the codebase, datasets, and resources used to benchmark volatility forecasting models as part of Xtreamly's exploration into crypto market dynamics. Our approach merges traditional econometric techniques (e.g., ARIMA, GARCH) with statistical evaluation to deliver robust and explainable predictions.

Whether you're a researcher, quant developer, or crypto enthusiast, this project serves as a solid foundation for your own experimentation or contributions.

---

## 📋 Requirements

1. **Python Version**: Python 3.11.3 (recommended for compatibility).
2. **Dependencies**: All required libraries are listed in the `requirements.txt` file.

## 🛠 Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Xtreamly-Team/xtreamly-backtesting.git
   cd xtreamly-backtesting
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Optional: Configure environment variables in a `.env` file (if needed for additional customizations).

## 📚 Documentation

This repository includes historical **ETH** and **BTC** datasets located in the `data/` directory. These datasets are preprocessed for ready-to-use computations.

If the datasets are too large to store directly in the repo, see the `docs/` folder for download instructions and data sourcing.

---

## 📈 Available Models

Implemented volatility forecasting models can be found in the `volatility/` directory:

- **ARIMA** – Autoregressive Integrated Moving Average (`volatility/model-arima.py`)
- **HAR** – Heterogeneous Autoregressive Model (`volatility/model-har.py`)
- **ARCH Family** – Including GARCH, EGARCH, etc. (`volatility/model-arch.py`)

Each model is modular with example usage included. Other models may be added in future updates.

---

## 📊 Data

All data used in the experiments is located in the `data/` directory or documented in the `docs/` folder for external access. Datasets are cleaned and ready for use with the provided models.

---

## 🤝 Contact

Have questions, suggestions, or collaboration ideas? Feel free to:

- Open an [issue on GitHub](https://github.com/Xtreamly-Team)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
