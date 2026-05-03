```markdown
# Tree Survival Game - Benchmark v1

## 📋 Benchmark Overview

This benchmark evaluates different tree selection policies in a survival game scenario where all trees have equal size, testing the ability to identify beneficial shape/label patterns rather than simply selecting the largest tree.

## 🎯 Benchmark Configuration

### Core Parameters
```python
label_count = 3
min_size = 6      # All trees have identical size
max_size = 6      # Eliminates "pick largest" trivial advantage
attempts_per_move = 5  # Small candidate pool per move
max_steps = 300    # Game length
episodes = 50      # Number of games per policy
```

### Environment Setup
- All trees have identical size (6)
- Only 5 candidate trees per move selection
- 3 possible labels for tree patterns
- 300-step game duration
- 50 episodes for statistical significance

## 📊 Baseline Results

| Policy | Average Score | Performance vs Random |
|--------|---------------|----------------------|
| Random | 75.66 | Baseline (100%) |
| Largest Tree | 67.82 | -10.4% |
| Heuristic | 227.68 | +201% |

**Key Insight**: When all candidates have the same size, "largest" selection loses its trivial advantage, and heuristic selection demonstrates meaningful pattern recognition capability.

## 🤖 Agent Policies

### 1. Random Selection (`choose_random`)
```python
def choose_random(valid_candidates):
    return random.choice(valid_candidates)
```

### 2. Largest Tree Selection (`choose_largest`)
```python
def choose_largest(valid_candidates):
    return max(valid_candidates, key=lambda tree: tree.size)
```

### 3. Heuristic Selection (`choose_heuristic`)
```python
def choose_heuristic(valid_candidates):
    return max(valid_candidates, key=tree_score)
```

### 4. Heuristic with Epsilon-Greedy Exploration (`choose_heuristic_epsilon`)
```python
def choose_heuristic_epsilon(valid_candidates):
    if random.random() < 0.10:  # 10% exploration
        return random.choice(valid_candidates)
    return max(valid_candidates, key=tree_score)
```

## 🔬 Experimental Design

### Why This Benchmark Matters
- **Level Playing Field**: Equal tree sizes remove trivial "pick largest" advantage
- **Pattern Recognition**: Tests ability to identify beneficial shape/label combinations
- **Strategic Depth**: Small candidate pool (5) forces meaningful choices
- **ML Relevance**: Clean baseline for training models to beat heuristic performance

### Success Criteria
A successful ML model should:
- Beat heuristic average of 227.68
- Demonstrate consistent performance across episodes
- Show improvement over random exploration

## 🚀 Next Steps

### Immediate Experiments
1. **Test Epsilon-Greedy**: Evaluate if 10% random exploration improves heuristic performance
2. **Feature Engineering**: Identify which tree characteristics contribute most to survival
3. **Model Training**: Develop ML models to predict optimal tree selection

### Future Enhancements
- Add more complex tree patterns
- Introduce dynamic environments
- Test transfer learning across different configurations
- Implement reinforcement learning approaches

## 📈 Performance Interpretation

### Current Understanding
- **Random**: Baseline performance (75.66)
- **Largest**: Actually performs worse than random when sizes are equal (-10.4%)
- **Heuristic**: Significant improvement (+201%) demonstrates meaningful pattern recognition

### ML Readiness
This benchmark provides:
- Clear target: Beat 227.68 average score
- Clean comparison: No trivial advantages
- Reproducible setup: Fixed parameters
- Statistical significance: 50 episodes

## 🛠️ Implementation Notes

### Code Structure
```python
agents = {
    "random": choose_random,
    "largest": choose_largest,
    "heuristic": choose_heuristic,
    "heuristic_eps": choose_heuristic_epsilon,
}

# Worker configuration
chooser_map = agents.copy()
```

### Running the Benchmark
```bash
python benchmark.py --min_size 6 --max_size 6 --attempts 5 --steps 300 --episodes 50
```

## 📚 References

This benchmark follows ML research best practices for:
- Reproducible experimental setup
- Clear baseline establishment
- Meaningful performance metrics
- Clean separation of trivial vs non-trivial advantages

---

*Last Updated: [Current Date]*
*Benchmark Version: v1*
```

This Markdown document provides a comprehensive overview of your benchmark with:

1. **Clear structure** following ML research documentation best practices
2. **Reproducible configuration** with all parameters explicitly defined
3. **Performance metrics** with baseline comparisons
4. **Experimental rationale** explaining why this setup is meaningful
5. **Next steps** for ML development
6. **Implementation details** for code reference

You can copy this into a `.md` file and use it as your benchmark documentation. Would you like me to make any adjustments to this format?