# Testing Framework

This directory contains comprehensive tests for the Fake Naira Detection system.

## Test Structure

```
tests/
├── README.md                    # This file
├── unit/                        # Unit tests
│   ├── test_detection_service.py
│   ├── test_image_utils.py
│   ├── test_model_utils.py
│   └── test_auth.py
├── integration/                 # Integration tests
│   ├── test_api_endpoints.py
│   ├── test_database.py
│   └── test_model_integration.py
├── e2e/                        # End-to-end tests
│   ├── test_upload_flow.py
│   ├── test_detection_flow.py
│   └── test_user_flow.py
├── fixtures/                    # Test fixtures and data
│   ├── sample_images/
│   ├── test_data.json
│   └── mock_responses.json
└── conftest.py                 # Pytest configuration
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual functions and methods
- **Scope**: Single components in isolation
- **Speed**: Fast execution (< 1 second per test)
- **Dependencies**: Minimal external dependencies

### Integration Tests
- **Purpose**: Test interaction between components
- **Scope**: Multiple components working together
- **Speed**: Medium execution (1-10 seconds per test)
- **Dependencies**: Database, external services

### End-to-End Tests
- **Purpose**: Test complete user workflows
- **Scope**: Full application stack
- **Speed**: Slow execution (10+ seconds per test)
- **Dependencies**: Full application, browser automation

## Running Tests

### Run All Tests
```bash
# From project root
pytest tests/

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# End-to-end tests only
pytest tests/e2e/
```

### Run Specific Tests
```bash
# Specific test file
pytest tests/unit/test_detection_service.py

# Specific test function
pytest tests/unit/test_detection_service.py::test_detect_counterfeit

# Tests matching pattern
pytest -k "test_detect"
```

## Test Configuration

### Pytest Configuration (conftest.py)
- Fixtures for common test data
- Database setup/teardown
- Mock configurations
- Test environment setup

### Environment Variables
- `TEST_DATABASE_URL`: Test database connection
- `TEST_MODEL_PATH`: Path to test model
- `TEST_IMAGE_PATH`: Path to test images

## Test Data

### Sample Images
- Located in `fixtures/sample_images/`
- Authentic naira note samples
- Counterfeit naira note samples
- Various denominations and conditions

### Mock Data
- `test_data.json`: Structured test data
- `mock_responses.json`: API response mocks
- Database fixtures for consistent testing

## Coverage Requirements

### Minimum Coverage Targets
- **Overall**: 90%
- **Critical Components**: 95%
- **API Endpoints**: 100%
- **ML Models**: 85%

### Coverage Reports
- HTML report: `htmlcov/index.html`
- Terminal output with coverage summary
- Coverage badges in README

## Test Best Practices

### Writing Tests
1. **Arrange-Act-Assert**: Clear test structure
2. **Descriptive Names**: Test names should describe what they test
3. **Single Responsibility**: One test, one assertion
4. **Independent Tests**: Tests should not depend on each other
5. **Fast Execution**: Keep tests fast and efficient

### Test Data Management
1. **Fixtures**: Use pytest fixtures for common data
2. **Cleanup**: Always clean up test data
3. **Isolation**: Tests should not affect each other
4. **Realistic Data**: Use realistic test data

### Mocking
1. **External Dependencies**: Mock external services
2. **File I/O**: Mock file operations when possible
3. **Network Calls**: Mock API calls and network requests
4. **Database**: Use test database for integration tests

## Continuous Integration

### GitHub Actions
- Run tests on every push
- Run tests on pull requests
- Generate coverage reports
- Upload test results

### Test Environments
- **Development**: Local testing
- **CI/CD**: Automated testing pipeline
- **Staging**: Pre-production testing
- **Production**: Smoke tests only

## Performance Testing

### Load Testing
- API endpoint performance
- Concurrent user handling
- Database query performance
- Model inference speed

### Stress Testing
- High volume image processing
- Memory usage under load
- System stability
- Error handling

## Security Testing

### Authentication Tests
- Login/logout functionality
- Token validation
- Permission checks
- Session management

### Input Validation
- File upload security
- Image format validation
- Size limit enforcement
- Malicious input handling

## Debugging Tests

### Common Issues
1. **Import Errors**: Check Python path and dependencies
2. **Database Issues**: Verify test database setup
3. **File Paths**: Use absolute paths for test files
4. **Async Tests**: Proper async/await usage

### Debug Commands
```bash
# Run with verbose output
pytest -v tests/

# Run with print statements
pytest -s tests/

# Run specific test with debugging
pytest --pdb tests/unit/test_detection_service.py

# Run with logging
pytest --log-cli-level=DEBUG tests/
```

## Test Maintenance

### Regular Tasks
- Update test data regularly
- Review and update test coverage
- Refactor tests for better maintainability
- Remove obsolete tests

### Test Documentation
- Keep test documentation up to date
- Document test data sources
- Explain complex test scenarios
- Maintain test runbooks

## Contact

For questions about testing or to report test issues:
- **Test Team**: tests@fakenairadetection.com
- **Technical Support**: support@fakenairadetection.com
