import sys

from PyQt6.QtWidgets import (QApplication, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QScrollArea,
                             QTextEdit, QVBoxLayout, QWidget)


class TradingUI(QWidget):
    def __init__(self, main_app_logic=None):  # Add main_app_logic
        super().__init__()
        self.main_app_logic = (
            main_app_logic  # Store a reference to the main application logic
        )
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("AI Options Trading Agent")
        self.setGeometry(100, 100, 800, 700)  # Increased size

        # Main layout
        main_layout = QVBoxLayout()

        # --- Operational Mode Display ---
        self.mode_display_label = QLabel("Mode: N/A")
        main_layout.addWidget(self.mode_display_label)

        # --- Input Section ---
        input_group = QGroupBox("Inputs")
        form_layout = QFormLayout()
        self.symbol_input = QLineEdit("AAPL")  # Default to AAPL
        self.capital_input = QLineEdit("100000")  # Default capital
        self.strategy_input = QLineEdit(
            "Simple Call Buyer")  # Default strategy changed
        self.proba_threshold_input = QLineEdit(
            "0.5")  # Default probability threshold
        self.data_period_input = QLineEdit("2y")  # Default data period (2 years)
        self.data_interval_input = QLineEdit("1d")  # Default data interval (daily)
        form_layout.addRow(QLabel("Stock Symbol:"), self.symbol_input)
        form_layout.addRow(QLabel("Initial Capital:"), self.capital_input)
        form_layout.addRow(QLabel("Strategy:"), self.strategy_input)
        form_layout.addRow(
            QLabel("ML Prob. Threshold:"), self.proba_threshold_input
        )  # Probability threshold input
        form_layout.addRow(
            QLabel("Data Period (1d,5d,1mo,3mo,6mo,1y,2y,5y,max):"), self.data_period_input
        )  # Data period input
        form_layout.addRow(
            QLabel("Data Interval (1m,2m,5m,15m,30m,60m,1h,1d,1wk,1mo):"), self.data_interval_input
        )  # Data interval input
        input_group.setLayout(form_layout)
        main_layout.addWidget(input_group)

        # --- Control Buttons ---
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Agent")
        if self.main_app_logic:  # Connect button if logic is provided
            self.run_button.clicked.connect(
                self.main_app_logic.run_trading_logic_from_ui
            )
        self.clear_button = QPushButton("Clear Logs")
        self.clear_button.clicked.connect(self.clear_all_outputs)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)
        main_layout.addLayout(button_layout)

        # --- Output Sections ---
        output_layout = QHBoxLayout()

        # Left side outputs (Market Data, Indicators)
        left_output_layout = QVBoxLayout()

        self.market_data_display = QTextEdit()
        self.market_data_display.setReadOnly(True)
        self.market_data_display.setPlaceholderText(
            "Market Data will appear here...")
        left_output_layout.addWidget(QLabel("Market Data & Indicators:"))
        left_output_layout.addWidget(self.market_data_display)

        self.ml_output_display = QTextEdit()
        self.ml_output_display.setReadOnly(True)
        self.ml_output_display.setPlaceholderText(
            "ML Model Output will appear here...")
        left_output_layout.addWidget(QLabel("ML Model Output:"))
        left_output_layout.addWidget(self.ml_output_display)

        output_layout.addLayout(left_output_layout)

        # Right side outputs (Strategy, Risk, Trades, General Log)
        right_output_layout = QVBoxLayout()

        self.strategy_output_display = QTextEdit()
        self.strategy_output_display.setReadOnly(True)
        self.strategy_output_display.setPlaceholderText(
            "Strategy Actions will appear here..."
        )
        right_output_layout.addWidget(QLabel("Strategy & Trade Execution:"))
        right_output_layout.addWidget(self.strategy_output_display)

        self.risk_output_display = QTextEdit()
        self.risk_output_display.setReadOnly(True)
        self.risk_output_display.setPlaceholderText(
            "Risk Assessment will appear here..."
        )
        right_output_layout.addWidget(QLabel("Risk Management:"))
        right_output_layout.addWidget(self.risk_output_display)

        output_layout.addLayout(right_output_layout)
        main_layout.addLayout(output_layout)

        # --- General Log / Status ---
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText(
            "Agent logs and status messages will appear here..."
        )
        main_layout.addWidget(QLabel("Agent Log:"))
        main_layout.addWidget(self.log_display)

        self.status_label = QLabel(
            "Welcome to the AI Options Trading Agent!"
        )  # Renamed from self.status
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

    def set_default_proba_threshold(self, threshold_value: float):
        """Sets the default value for the ML probability threshold input field."""
        if threshold_value is not None:
            self.proba_threshold_input.setText(str(threshold_value))
            self.log_message(
                f"CLI provided default probability threshold: {threshold_value}"
            )

    def set_operational_mode_display(self, mode: str):
        """Updates the UI to show the current operational mode."""
        self.mode_display_label.setText(f"Mode: {mode.upper()}")
        self.log_message(f"Operational mode set to: {mode.upper()}")

    def get_inputs(self):
        return {
            "symbol": self.symbol_input.text().upper(),
            "capital": (
                float(self.capital_input.text())
                if self.capital_input.text()
                else 100000.0
            ),
            "strategy_name": self.strategy_input.text(),
            "proba_threshold": (
                float(self.proba_threshold_input.text())
                if self.proba_threshold_input.text()
                else 0.5
            ),  # Retrieve threshold
            "data_period": (
                self.data_period_input.text()
                if self.data_period_input.text()
                else "2y"
            ),  # Retrieve data period
            "data_interval": (
                self.data_interval_input.text()
                if self.data_interval_input.text()
                else "1d"
            ),  # Retrieve data interval
        }

    def update_status(self, text):
        self.status_label.setText(text)  # Use renamed status_label
        self.log_message(f"STATUS: {text}")

    def log_message(self, text):
        self.log_display.append(str(text))

    def display_market_data(self, data_dict):
        # data_dict can contain various pandas series/dataframes
        # For now, just converting to string. Could be formatted better.
        display_text = ""
        if data_dict.get("historical_data") is not None:
            display_text += "--- Historical Data (tail) ---\n"
            display_text += str(data_dict["historical_data"].tail()) + "\n\n"
        if data_dict.get("ma_data") is not None:
            display_text += "--- 20-day MA (tail) ---\n"
            display_text += str(data_dict["ma_data"].tail()) + "\n\n"
        if data_dict.get("rsi_data") is not None:
            display_text += "--- RSI (tail) ---\n"
            display_text += str(data_dict["rsi_data"].tail()) + "\n\n"
        if (
            data_dict.get("bollinger_upper") is not None
            and data_dict.get("bollinger_lower") is not None
        ):
            display_text += "--- Bollinger Bands (tail) ---\n"
            display_text += "Upper:\n" + \
                str(data_dict["bollinger_upper"].tail()) + "\n"
            display_text += (
                "Lower:\n" + str(data_dict["bollinger_lower"].tail()) + "\n\n"
            )
        if data_dict.get("volatility_data") is not None:
            display_text += "--- Volatility (tail) ---\n"
            display_text += str(data_dict["volatility_data"].tail()) + "\n"

        self.market_data_display.setPlainText(
            display_text if display_text else "No market data to display."
        )

    def display_ml_output(self, accuracy, metrics=None):
        accuracy_percentage = accuracy * 100
        text = f"Model Accuracy: {accuracy_percentage:.1f}%\n\n"
        
        if not metrics:
            self.ml_output_display.setPlainText(text)
            return
            
        # Handle simple status messages
        if "status" in metrics and len(metrics) == 1:
            text += f"Status: {metrics['status']}\n"
            self.ml_output_display.setPlainText(text)
            return
            
        # Handle structured metrics (cv and holdout)
        if "cv" in metrics and "holdout" in metrics:
            # Display cross-validation metrics
            text += "=== Cross-Validation Metrics ===\n"
            cv_metrics = metrics["cv"]
            for key, value in cv_metrics.items():
                if isinstance(value, float):
                    text += f"  {key.replace('_', ' ').title()}: {value*100:.1f}%\n"
                else:
                    text += f"  {key.replace('_', ' ').title()}: {value}\n"
            
            # Display holdout metrics
            text += "\n=== Hold-out Test Metrics ===\n"
            holdout_metrics = metrics["holdout"]
            for key, value in holdout_metrics.items():
                if isinstance(value, float):
                    text += f"  {key.replace('_', ' ').title()}: {value*100:.1f}%\n"
                else:
                    text += f"  {key.replace('_', ' ').title()}: {value}\n"
        # Handle flat metrics dictionary (legacy format)
        else:
            text += "=== Model Metrics ===\n"
            for key, value in metrics.items():
                if key == "precision" or key == "precision_score":
                    text += f"  Precision: {value*100:.1f}% of signals were correct.\n"
                elif key == "recall" or key == "recall_score":
                    text += f"  Recall: Caught {value*100:.1f}% of actual upward moves.\n"
                elif key == "f1" or key == "f1_score":
                    text += f"  F1-Score: {value*100:.1f}% (harmonic mean of precision & recall).\n"
                elif key == "roc_auc":
                    text += f"  ROC-AUC: {value*100:.1f}% (area under ROC curve).\n"
                elif key == "status":
                    text += f"  Status: {value}\n"
                else:
                    if isinstance(value, float):
                        text += f"  {key.replace('_', ' ').title()}: {value*100:.1f}%\n"
                    else:
                        text += f"  {key.replace('_', ' ').title()}: {value}\n"
        
        self.ml_output_display.setPlainText(text)

    def display_strategy_action(self, action):
        self.strategy_output_display.append(f"Strategy Action: {action}")

    def display_trade_execution(self, details):
        self.strategy_output_display.append(f"Trade Executed: {details}")

    def display_risk_assessment(self, assessment):
        self.risk_output_display.setPlainText(f"Risk Assessment: {assessment}")

    def clear_all_outputs(self):
        self.market_data_display.clear()
        self.ml_output_display.clear()
        self.strategy_output_display.clear()
        self.risk_output_display.clear()
        self.log_display.clear()
        self.update_status("Logs cleared. Ready for new run.")

    def launch(self):  # Keep this for standalone testing if needed
        # This method should ideally be called by main.py, not internally like this
        # For now, we'll keep it but the main app will control the QApplication lifecycle
        app = QApplication.instance()  # Check if an app instance exists
        if not app:  # Create one if not
            app = QApplication(sys.argv)

        self.show()
        # sys.exit(app.exec()) # main.py should handle app.exec()
        return app  # Return app instance for main.py to manage


if __name__ == "__main__":
    # This is for testing the UI independently
    app = QApplication(sys.argv)
    ui = TradingUI()
    ui.show()
    sys.exit(app.exec())
