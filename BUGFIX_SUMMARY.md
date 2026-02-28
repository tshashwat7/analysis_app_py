# Bug Fix Summary: GENERIC Setup Dominance

## Problem
The GENERIC setup was always winning over specific setups (MOMENTUM_BREAKOUT, TREND_FOLLOWING, REVERSAL_ST_FLIP_UP, etc.) regardless of technical conditions.

## Root Cause Analysis

### Bug #1 & #2: Operator Splitting Off-by-One Error
**File:** `config/config_resolver.py` Line 115

The condition evaluator had an off-by-one error when splitting conditions by operators. The regex pattern `\s{operator}\s` matches the space **before** the operator, but the code failed to account for this when extracting the right operand.

**Example - BEFORE (Broken):**
```
Condition: supertrendSignal == 'Bullish'
Pattern: \s==\s matches at position 17 (the space before ==)
match_pos = 17 - 1 = 16
left = condition[:16].strip() = "supertrendSignal" ✓
right = condition[16 + 2:] = condition[18:] = "= 'Bullish'" ❌
```

The right operand included the extra `=`, causing `_parse_right_operand("= 'Bullish'")` to fail and return `None`.

**AFTER (Fixed):**
```
right = condition[16 + 1 + 2:] = condition[19:] = "'Bullish'" ✓
```

### Bug #3: MIN_FIT_SCORE Never Enforced
**File:** `config/config_resolver.py` Line 43 & 1162

`MIN_FIT_SCORE = 10.0` was defined but never enforced as a filter. Low-quality setups weren't rejected, allowing invalid candidates to win through composite scoring.

### Bug #4: Silent Rejection Logging
No visibility into why specific setups were being rejected during the selection process, making debugging impossible.

## Fixes Applied

### Fix #1: Operator Splitting (Line 115)
```python
# BEFORE:
right = condition[match_pos + len(op_str):].strip()

# AFTER:
right = condition[match_pos + 1 + len(op_str):].strip()  # +1 to skip the space before operator
```

This single character fix (`+1`) repairs ALL operator splits for all operators (`>=`, `<=`, `>`, `<`, `==`, `!=`).

**Impact:** 
- `rsi >= 55` now correctly returns `True` when rsi=78.28
- `supertrendSignal == 'Bullish'` now correctly returns `True`
- `macdCross == 'Bullish'` works correctly
- `bbWidth < 0.5` works correctly
- All numeric and string comparisons now work

### Fix #2: MIN_FIT_SCORE Enforcement (Lines 1165-1170)
```python
if fit_score < MIN_FIT_SCORE:
    rejected.append({
        "type": setup_name,
        "reason": f"fit_score_too_low ({fit_score:.1f} < {MIN_FIT_SCORE})"
    })
    continue
```

**Impact:** Low-quality setups are now filtered out before ranking, preventing invalid matches from winning.

### Fix #3: Rejection Logging (Lines 1191-1201)
```python
if self.logger.isEnabledFor(logging.DEBUG):
    for r in rejected:
        self.logger.debug(f"  ↳ Setup rejected: {r['type']} | reason={r['reason']}")
    for c in ranked:
        self.logger.debug(
            f"  ↳ Setup candidate: {c['type']} | priority={c['priority']} | "
            f"fit={c['fit_score']:.1f} | composite={c['composite_score']:.1f}"
        )
```

**Impact:** Debug logs now show exactly why setups were rejected and which candidates made the cut.

## Verification

Tested condition parsing with both string and numeric comparisons:
- ✅ `supertrendSignal == 'Bullish'` → correctly parses to left='supertrendSignal', right=''Bullish''
- ✅ `rsi >= 55` → correctly parses to left='rsi', right='55'

## Expected Behavior After Fix

1. **Specific setups win when conditions match** - MOMENTUM_BREAKOUT, TREND_FOLLOWING, etc. will now be selected when their technical conditions are truly met.

2. **GENERIC becomes the fallback** - GENERIC will only win when no specific setup's conditions are met or when fit scores are below MIN_FIT_SCORE.

3. **Better debugging** - Debug logs will show:
   ```
   Setup rejected: MOMENTUM_BREAKOUT | reason=rsi_too_low
   Setup rejected: TREND_FOLLOWING | reason=fit_score_too_low
   Setup candidate: REVERSAL_ST_FLIP_UP | priority=9 | fit=65.3 | composite=31.8
   ```

## Testing the Fix

Run the analyzer on any stock ticker. You should now see:
- Different setups selected for different market conditions
- Setup selection based on actual technical metrics
- Debug logs showing why each setup was accepted or rejected

Example expected output:
```
INFO | SETUP SELECTED: TREND_FOLLOWING | Priority=15 | Fit=72.5 | Composite=45.1
INFO | SETUP SELECTED: MOMENTUM_BREAKOUT | Priority=8 | Fit=58.3 | Composite=29.4
```

(Instead of always seeing GENERIC)
