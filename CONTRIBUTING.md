# Contributing to AI Trading System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)
- Relevant logs or error messages

### Suggesting Features

Feature requests are welcome! Please:
- Check if the feature already exists or is planned
- Describe the use case and benefits
- Provide examples if possible

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow the code style (PEP 8)
   - Add docstrings to functions/classes
   - Update tests if applicable
4. **Test your changes**
   ```bash
   python main.py mock      # Test with mock data
   python main.py backtest  # Run backtest
   ```
5. **Commit with clear messages**
   ```bash
   git commit -m "feat: add new indicator for momentum detection"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

## 📝 Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Maximum line length: 100 characters
- Use meaningful variable names
- Add docstrings to all public functions/classes

Example:
```python
def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    stop_distance: float,
) -> int:
    """
    Calculate position size based on risk parameters.

    Args:
        capital: Total account capital
        risk_per_trade: Risk percentage (e.g., 0.01 for 1%)
        stop_distance: Distance to stop loss in price units

    Returns:
        Position size (number of contracts)
    """
    risk_amount = capital * risk_per_trade
    return max(1, int(risk_amount / stop_distance))
```

## 🧪 Testing

Before submitting a PR:
- Run mock mode to verify basic functionality
- Run backtest mode to ensure strategies work
- Test with edge cases (empty data, missing features, etc.)

## 📋 Commit Message Convention

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Examples:
```
feat: add LSTM model for time series prediction
fix: resolve VWAP calculation error with missing data
docs: update README with deployment instructions
refactor: simplify trade scoring logic
```

## 🎯 Areas for Contribution

### High Priority
- [ ] Unit tests for all modules
- [ ] Integration tests for full pipeline
- [ ] Dashboard/UI (Streamlit or Gradana)
- [ ] Telegram notifications
- [ ] Performance optimization

### Medium Priority
- [ ] Additional technical indicators
- [ ] More strategy implementations
- [ ] Support for other brokers (Upstox, Angel One)
- [ ] Multi-symbol portfolio optimization
- [ ] Advanced ML models (LSTM, Transformers)

### Low Priority
- [ ] Backtesting improvements (slippage, commissions)
- [ ] Paper trading mode
- [ ] Historical data replay
- [ ] Strategy parameter optimization

## 🔍 Code Review Process

All PRs will be reviewed for:
- Code quality and style
- Test coverage
- Documentation
- Performance impact
- Security considerations

## ⚠️ Important Guidelines

### Security
- Never commit API keys or credentials
- Use environment variables for sensitive data
- Review `.gitignore` before committing

### Data
- Don't commit large datasets
- Don't commit trained model files (*.pkl)
- Use mock data for examples

### Performance
- Profile code for bottlenecks
- Avoid unnecessary database queries
- Use batch operations where possible

## 📚 Resources

- [Project Documentation](docs/)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Product Vision](docs/Product_vision.md)

## 💬 Communication

- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for questions and ideas
- Be respectful and constructive in all interactions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the AI Trading System! 🚀
