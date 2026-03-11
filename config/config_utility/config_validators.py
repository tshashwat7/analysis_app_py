# config/config_validators.py
"""
Configuration Validators & Builders for V2 SMRT Configs
Provides validation, schema checking, and config building utilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================
# VALIDATION RESULT CLASSES
# ============================================================

class ValidationLevel(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Blocks execution
    WARNING = "warning"  # Should fix but won't block
    INFO = "info"        # Nice to know


@dataclass
class ValidationIssue:
    """Single validation issue."""
    level: ValidationLevel
    path: str
    message: str
    actual: Any = None
    expected: Any = None
    
    def __str__(self) -> str:
        msg = f"[{self.level.value.upper()}] {self.path}: {self.message}"
        if self.actual is not None:
            msg += f" (got: {self.actual})"
        if self.expected is not None:
            msg += f" (expected: {self.expected})"
        return msg


@dataclass
class ValidationReport:
    """Complete validation report."""
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    
    def add(self, issue: ValidationIssue):
        """Add issue to appropriate category."""
        if issue.level == ValidationLevel.ERROR:
            self.errors.append(issue)
        elif issue.level == ValidationLevel.WARNING:
            self.warnings.append(issue)
        else:
            self.info.append(issue)
    
    @property
    def is_valid(self) -> bool:
        """True if no errors."""
        return len(self.errors) == 0
    
    @property
    def total_issues(self) -> int:
        """Total count of all issues."""
        return len(self.errors) + len(self.warnings) + len(self.info)
    
    def print_report(self, show_info: bool = False):
        """Print formatted validation report."""
        print("\n" + "="*70)
        print("CONFIG VALIDATION REPORT")
        print("="*70)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for issue in self.errors:
                print(f"  {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for issue in self.warnings:
                print(f"  {issue}")
        
        if show_info and self.info:
            print(f"\nℹ️  INFO ({len(self.info)}):")
            for issue in self.info:
                print(f"  {issue}")
        
        print("\n" + "="*70)
        if self.is_valid:
            print("✅ VALIDATION PASSED")
        else:
            print(f"❌ VALIDATION FAILED - {len(self.errors)} error(s)")
        print("="*70 + "\n")


# ============================================================
# CONFIG VALIDATOR
# ============================================================

class ConfigValidator:
    """
    Validates SMRT V2 configurations for completeness and correctness.
    
    Checks:
    - Required sections exist
    - Data types are correct
    - Value ranges are valid
    - Cross-references are consistent
    - Inheritance chains are complete
    """
    
    # Required global sections
    REQUIRED_GLOBAL = [
        "system",
        "position_sizing",
        "technical_weights",
        "fundamental_weights",
        "calculation_engine",
        "pattern_entry_rules",
        "pattern_invalidation",
        "time_estimation",
        "strategy_classification"
    ]
    
    # Required horizon sections
    REQUIRED_HORIZON = [
        "timeframe",
        "description",
        "indicators",
        "execution",
        "risk_management",
        "gates",
        "moving_averages"
    ]
    
    # Required indicator parameters
    REQUIRED_INDICATORS = [
        "adx_period",
        "atr_period",
        "rsi_period",
        "stochastic"
    ]
    
    def __init__(self, config: Dict):
        """
        Initialize validator with config.
        
        Args:
            config: Master config dict
        """
        self.config = config
        self.report = ValidationReport()
    
    def validate_all(self) -> ValidationReport:
        """
        Run all validation checks.
        
        Returns:
            ValidationReport with all issues
        """
        logger.info("Starting comprehensive config validation...")
        
        # Structure checks
        self._validate_structure()
        
        # Global section checks
        self._validate_global_sections()
        
        # Horizon checks
        self._validate_all_horizons()
        
        # Cross-validation
        self._validate_cross_references()
        self._validate_inheritance_chains()
        
        # Type and range checks
        self._validate_types_and_ranges()
        
        logger.info(f"Validation complete: {self.report.total_issues} issues found")
        return self.report
    
    # ============================================================
    # STRUCTURE VALIDATION
    # ============================================================
    
    def _validate_structure(self):
        """Check top-level structure."""
        if not isinstance(self.config, dict):
            self.report.add(ValidationIssue(
                level=ValidationLevel.ERROR,
                path="root",
                message="Config must be a dictionary"
            ))
            return
        
        # Check for global section
        if "global" not in self.config:
            self.report.add(ValidationIssue(
                level=ValidationLevel.ERROR,
                path="global",
                message="Missing global section"
            ))
        
        # Check for horizons section
        if "horizons" not in self.config:
            self.report.add(ValidationIssue(
                level=ValidationLevel.ERROR,
                path="horizons",
                message="Missing horizons section"
            ))
        else:
            horizons = self.config["horizons"]
            if not isinstance(horizons, dict):
                self.report.add(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    path="horizons",
                    message="Horizons must be a dictionary"
                ))
            else:
                expected = {"intraday", "short_term", "long_term", "multibagger"}
                actual = set(horizons.keys())
                missing = expected - actual
                if missing:
                    self.report.add(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        path="horizons",
                        message=f"Missing horizons: {missing}"
                    ))
    
    def _validate_global_sections(self):
        """Validate global section completeness."""
        if "global" not in self.config:
            return  # Already reported in structure check
        
        global_config = self.config["global"]
        
        for section in self.REQUIRED_GLOBAL:
            if section not in global_config:
                self.report.add(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    path=f"global.{section}",
                    message=f"Missing global section: {section}"
                ))
    
    def _validate_all_horizons(self):
        """Validate each horizon configuration."""
        if "horizons" not in self.config:
            return
        
        for horizon, config in self.config["horizons"].items():
            self._validate_horizon(horizon, config)
    
    def _validate_horizon(self, horizon: str, config: Dict):
        """Validate single horizon configuration with inheritance awareness."""
        
        # These MUST exist in the horizon directly
        REQUIRED_DIRECT = ["timeframe", "description"]
        
        for section in REQUIRED_DIRECT:
            if section not in config:
                self.report.add(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    path=f"horizons.{horizon}.{section}",
                    message=f"Missing required direct field"
                ))
        
        # These CAN inherit from global
        INHERITABLE = ["indicators", "execution", "risk_management", "gates", "moving_averages", "momentum_thresholds", "volatility"]
        
        for section in INHERITABLE:
            # Check if it exists in horizon OR global
            in_horizon = section in config
            in_global = self.config.get("global", {}).get(section) is not None
            
            if not in_horizon and not in_global:
                self.report.add(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    path=f"horizons.{horizon}.{section}",
                    message=f"Missing section with no global fallback"
                ))
            elif not in_horizon:
                # It's OK - will inherit from global
                self.report.add(ValidationIssue(
                    level=ValidationLevel.INFO,
                    path=f"horizons.{horizon}.{section}",
                    message=f"Inheriting from global.{section}"
                ))
        
        # Validate sections that DO exist in the horizon
        # ✅ FIX: Add underscore prefix to method names
        if "indicators" in config:
            self._validate_indicators(f"horizons.{horizon}.indicators", config["indicators"])
        
        if "execution" in config:
            self._validate_execution(f"horizons.{horizon}.execution", config["execution"])
        
        if "riskmanagement" in config:
            self._validate_risk_management(f"horizons.{horizon}.riskmanagement", config["riskmanagement"])
        
        if "gates" in config:
            self._validate_gates(f"horizons.{horizon}.gates", config["gates"])

    # ============================================================
    # SECTION-SPECIFIC VALIDATION
    # ============================================================
    
    def _validate_indicators(self, path: str, indicators: Dict):
        """Validate indicator configuration."""
        for param in self.REQUIRED_INDICATORS:
            if param not in indicators:
                self.report.add(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    path=f"{path}.{param}",
                    message=f"Missing required indicator parameter"
                ))
        
        # Validate periods are positive integers
        period_keys = ["adx_period", "atr_period", "rsi_period"]
        for key in period_keys:
            if key in indicators:
                value = indicators[key]
                if not isinstance(value, int) or value < 1:
                    self.report.add(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        path=f"{path}.{key}",
                        message="Period must be positive integer",
                        actual=value,
                        expected="> 0"
                    ))
        
        # Validate stochastic
        if "stochastic" in indicators:
            stoch = indicators["stochastic"]
            required = ["k", "d", "smooth"]
            for param in required:
                if param not in stoch:
                    self.report.add(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        path=f"{path}.stochastic.{param}",
                        message="Missing stochastic parameter"
                    ))
    
    def _validate_execution(self, path: str, execution: Dict):
        """Validate execution parameters."""
        required = [
            "stop_loss_atr_mult",
            "target_atr_mult",
            "max_hold_candles",
            "base_hold_days"
        ]
        
        for param in required:
            if param not in execution:
                self.report.add(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    path=f"{path}.{param}",
                    message="Missing execution parameter"
                ))
        
        # Validate multipliers are positive
        if "stop_loss_atr_mult" in execution:
            val = execution["stop_loss_atr_mult"]
            if not isinstance(val, (int, float)) or val <= 0:
                self.report.add(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    path=f"{path}.stop_loss_atr_mult",
                    message="Stop loss multiplier must be positive",
                    actual=val
                ))
        
        # Validate target > stop loss
        if "target_atr_mult" in execution and "stop_loss_atr_mult" in execution:
            target = execution["target_atr_mult"]
            sl = execution["stop_loss_atr_mult"]
            if target <= sl:
                self.report.add(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    path=f"{path}.target_atr_mult",
                    message="Target should be greater than stop loss",
                    actual=f"T:{target} vs SL:{sl}"
                ))
    
    def _validate_risk_management(self, path: str, risk: Dict):
        """Validate risk management parameters."""
        # Check position sizing
        if "max_position_pct" in risk:
            val = risk["max_position_pct"]
            if not (0 < val <= 1.0):
                self.report.add(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    path=f"{path}.max_position_pct",
                    message="Position size must be between 0 and 1",
                    actual=val,
                    expected="0 < x <= 1.0"
                ))
        
        # Check RR ratio
        if "min_rr_ratio" in risk:
            val = risk["min_rr_ratio"]
            if val < 1.0:
                self.report.add(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    path=f"{path}.min_rr_ratio",
                    message="Min RR ratio < 1.0 is risky",
                    actual=val,
                    expected=">= 1.0"
                ))
    
    def _validate_gates(self, path: str, gates: Dict):
        """Validate entry gate thresholds."""
        # Check volatility bands
        if "volatility_bands_atr_pct" in gates:
            bands = gates["volatility_bands_atr_pct"]
            if not isinstance(bands, dict):
                self.report.add(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    path=f"{path}.volatility_bands_atr_pct",
                    message="Volatility bands must be a dict"
                ))
            else:
                required = ["min", "ideal", "max"]
                for key in required:
                    if key not in bands:
                        self.report.add(ValidationIssue(
                            level=ValidationLevel.ERROR,
                            path=f"{path}.volatility_bands_atr_pct.{key}",
                            message=f"Missing volatility band: {key}"
                        ))
                
                # Check ordering
                if all(k in bands for k in required):
                    if not (bands["min"] < bands["ideal"] < bands["max"]):
                        self.report.add(ValidationIssue(
                            level=ValidationLevel.ERROR,
                            path=f"{path}.volatility_bands_atr_pct",
                            message="Bands must be ordered: min < ideal < max",
                            actual=f"min:{bands['min']}, ideal:{bands['ideal']}, max:{bands['max']}"
                        ))
    
    # ============================================================
    # CROSS-VALIDATION
    # ============================================================
    
    def _validate_cross_references(self):
        """Validate references between sections."""
        # Check pattern entry rules reference valid patterns
        if "global" in self.config and "pattern_entry_rules" in self.config["global"]:
            rules = self.config["global"]["pattern_entry_rules"]
            for pattern, rule in rules.items():
                if "horizons" in rule:
                    for horizon in rule["horizons"]:
                        if horizon not in self.config.get("horizons", {}):
                            self.report.add(ValidationIssue(
                                level=ValidationLevel.WARNING,
                                path=f"global.pattern_entry_rules.{pattern}",
                                message=f"References non-existent horizon: {horizon}"
                            ))
    
    def _validate_inheritance_chains(self):
        """Validate inheritance chains are complete."""
        # For each horizon section, check if global fallback exists
        for horizon, h_config in self.config.get("horizons", {}).items():
            for section in self.REQUIRED_HORIZON:
                if section not in h_config:
                    # Check if global has it
                    if section not in self.config.get("global", {}):
                        self.report.add(ValidationIssue(
                            level=ValidationLevel.ERROR,
                            path=f"horizons.{horizon}.{section}",
                            message=f"Missing section with no global fallback"
                        ))
    
    def _validate_types_and_ranges(self):
        """Validate data types and value ranges."""
        # Validate all numeric thresholds
        self._check_numeric_range("gates.adx_min", 0, 100)
        self._check_numeric_range("gates.min_trend_strength", 0, 10)
        self._check_numeric_range("indicators.rsi_period", 2, 100)
        self._check_numeric_range("indicators.adx_period", 5, 50)
    
    def _check_numeric_range(self, path: str, min_val: float, max_val: float):
        """Helper to check if numeric value is in range."""
        for horizon in self.config.get("horizons", {}).keys():
            full_path = f"horizons.{horizon}.{path}"
            value = self._get_nested_value(self.config, full_path)
            if value is not None:
                if not (min_val <= value <= max_val):
                    self.report.add(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        path=full_path,
                        message=f"Value outside recommended range",
                        actual=value,
                        expected=f"{min_val} to {max_val}"
                    ))
    
    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get nested value from dict using dot path."""
        keys = path.split('.')
        current = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current


# ============================================================
# CONFIG BUILDER
# ============================================================

class ConfigBuilder:
    """
    Fluent builder for creating V2 SMRT configurations.
    
    Example:
        config = (ConfigBuilder()
            .add_horizon("intraday")
            .set_indicators(adx_period=10, atr_period=10)
            .set_execution(stop_loss_atr_mult=1.5, target_atr_mult=3.0)
            .build())
    """
    
    def __init__(self):
        self.config = {
            "global": {},
            "horizons": {}
        }
        self._current_horizon = None
    
    def add_horizon(self, name: str, timeframe: str = "1d", 
                   description: str = "") -> 'ConfigBuilder':
        """
        Add a new horizon.
        
        Args:
            name: Horizon name
            timeframe: Timeframe string (e.g., "5m", "1d")
            description: Human-readable description
            
        Returns:
            Self for chaining
        """
        self.config["horizons"][name] = {
            "timeframe": timeframe,
            "description": description,
            "indicators": {},
            "execution": {},
            "risk_management": {},
            "gates": {},
            "moving_averages": {}
        }
        self._current_horizon = name
        return self
    
    def set_indicators(self, **kwargs) -> 'ConfigBuilder':
        """
        Set indicator parameters for current horizon.
        
        Example:
            .set_indicators(adx_period=10, atr_period=10, rsi_period=9)
        """
        if self._current_horizon:
            self.config["horizons"][self._current_horizon]["indicators"].update(kwargs)
        return self
    
    def set_execution(self, **kwargs) -> 'ConfigBuilder':
        """Set execution parameters for current horizon."""
        if self._current_horizon:
            self.config["horizons"][self._current_horizon]["execution"].update(kwargs)
        return self
    
    def set_risk_management(self, **kwargs) -> 'ConfigBuilder':
        """Set risk management parameters for current horizon."""
        if self._current_horizon:
            self.config["horizons"][self._current_horizon]["risk_management"].update(kwargs)
        return self
    
    def set_gates(self, **kwargs) -> 'ConfigBuilder':
        """Set entry gate thresholds for current horizon."""
        if self._current_horizon:
            self.config["horizons"][self._current_horizon]["gates"].update(kwargs)
        return self
    
    def set_moving_averages(self, ma_type: str = "EMA", 
                           fast: int = 20, mid: int = 50, 
                           slow: int = 200) -> 'ConfigBuilder':
        """Set moving average configuration for current horizon."""
        if self._current_horizon:
            self.config["horizons"][self._current_horizon]["moving_averages"] = {
                "type": ma_type,
                "fast": fast,
                "mid": mid,
                "slow": slow
            }
        return self
    
    def add_global_section(self, section_name: str, data: Dict) -> 'ConfigBuilder':
        """Add a global section."""
        self.config["global"][section_name] = data
        return self
    
    def build(self) -> Dict:
        """
        Build and return the final config dict.
        
        Returns:
            Complete config dictionary
        """
        return self.config
    
    def validate_and_build(self) -> Tuple[Dict, ValidationReport]:
        """
        Build config and validate it.
        
        Returns:
            Tuple of (config, validation_report)
        """
        config = self.build()
        validator = ConfigValidator(config)
        report = validator.validate_all()
        return config, report


# ============================================================
# USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    # Example 1: Validate existing config
    print("="*70)
    print("EXAMPLE 1: Validating Master Config")
    print("="*70)
    
    from config.master_config import MASTER_CONFIG
    
    validator = ConfigValidator(MASTER_CONFIG)
    report = validator.validate_all()
    report.print_report(show_info=True)
    
    # Example 2: Build a new config
    print("\n" + "="*70)
    print("EXAMPLE 2: Building New Config")
    print("="*70)
    
    config, report = (ConfigBuilder()
        # Add intraday horizon
        .add_horizon("intraday", timeframe="5m", description="Fast scalping")
        .set_indicators(adx_period=10, atr_period=10, rsi_period=9)
        .set_execution(
            stop_loss_atr_mult=1.5,
            target_atr_mult=3.0,
            max_hold_candles=48,
            base_hold_days=1
        )
        .set_risk_management(
            max_position_pct=0.02,
            min_rr_ratio=2.0
        )
        .set_gates(
            adx_min=18,
            min_trend_strength=3.0,
            volatility_bands_atr_pct={"min": 0.3, "ideal": 3.0, "max": 5.0}
        )
        .set_moving_averages(ma_type="EMA", fast=20, mid=50, slow=200)
        
        # Add short-term horizon
        .add_horizon("short_term", timeframe="1d", description="Swing trading")
        .set_indicators(adx_period=14, atr_period=14, rsi_period=14)
        .set_execution(
            stop_loss_atr_mult=2.0,
            target_atr_mult=4.0,
            max_hold_candles=20,
            base_hold_days=10
        )
        .set_risk_management(
            max_position_pct=0.03,
            min_rr_ratio=2.0
        )
        
        # Build and validate
        .validate_and_build()
    )
    
    print("\n✓ Config built successfully")
    print(f"  Horizons: {list(config['horizons'].keys())}")
    
    report.print_report()
