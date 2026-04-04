"""Generate SVG diagrams for the GA Feature Selection course."""
import sys
sys.path.insert(0, '.')

from resources.graphics import process_flow, comparison_graphic, architecture_diagram, timeline, concept_map

BASE = "courses/genetic-algorithms-feature-selection/modules"

# 1. GA Lifecycle
ga_lifecycle = process_flow(
    steps=["Initialize\nPopulation", "Evaluate\nFitness", "Selection", "Crossover", "Mutation", "New\nGeneration"],
    colors=["mint", "blue", "amber", "lavender", "rose", "mint"],
    title="Genetic Algorithm Lifecycle"
)
ga_lifecycle.save(f"{BASE}/module_01_ga_fundamentals/guides/ga_lifecycle.svg")

# 2. Selection Methods
selection = comparison_graphic(
    left={"title": "Tournament Selection", "items": ["Pick k random individuals", "Best one wins", "Adjustable pressure via k"]},
    right={"title": "Roulette Wheel", "items": ["Probability proportional to fitness", "Spin wheel to select", "High fitness dominates"]},
    title="Selection Method Comparison"
)
selection.save(f"{BASE}/module_01_ga_fundamentals/guides/selection_methods.svg")

# 3. Crossover Types
crossover = process_flow(
    steps=["Single-Point", "Two-Point", "Uniform"],
    colors=["mint", "amber", "blue"],
    title="Crossover Operators"
)
crossover.save(f"{BASE}/module_01_ga_fundamentals/guides/crossover_types.svg")

# 4. Mutation Types
mutation = process_flow(
    steps=["Bit Flip", "Swap", "Scramble"],
    colors=["rose", "lavender", "amber"],
    title="Mutation Operators"
)
mutation.save(f"{BASE}/module_01_ga_fundamentals/guides/mutation_types.svg")

# 5. Fitness Landscape
fitness = concept_map(
    nodes=[
        {"id": "landscape", "label": "Fitness\nLandscape", "color": "blue"},
        {"id": "global", "label": "Global\nOptimum", "color": "mint"},
        {"id": "local", "label": "Local\nOptima", "color": "amber"},
        {"id": "plateau", "label": "Plateaus", "color": "gray"},
        {"id": "diversity", "label": "Population\nDiversity", "color": "lavender"},
    ],
    edges=[
        {"from": "landscape", "to": "global", "label": "contains"},
        {"from": "landscape", "to": "local", "label": "contains"},
        {"from": "landscape", "to": "plateau", "label": "contains"},
        {"from": "diversity", "to": "local", "label": "escapes"},
    ],
    title="Fitness Landscape Concepts"
)
fitness.save(f"{BASE}/module_02_fitness/guides/fitness_landscape.svg")

# 6. Feature Selection Pipeline
pipeline = architecture_diagram(
    layers=[
        {"name": "Data", "nodes": ["Raw Features", "Preprocessing"], "color": "gray"},
        {"name": "GA Engine", "nodes": ["Population", "Fitness Eval", "Selection", "Operators"], "color": "mint"},
        {"name": "Output", "nodes": ["Best Features", "Model Training", "Evaluation"], "color": "blue"},
    ],
    connections=[
        ("Preprocessing", "Population"),
        ("Population", "Fitness Eval"),
        ("Fitness Eval", "Selection"),
        ("Selection", "Operators"),
        ("Operators", "Population"),
        ("Best Features", "Model Training"),
        ("Model Training", "Evaluation"),
    ],
    title="GA Feature Selection Pipeline"
)
pipeline.save(f"{BASE}/module_00_foundations/guides/feature_selection_pipeline.svg")

# 7. Walk-Forward Timeline
wf = timeline(
    events=[
        {"label": "Train 1", "detail": "t0 to t1", "color": "mint"},
        {"label": "Test 1", "detail": "t1 to t2", "color": "blue"},
        {"label": "Train 2", "detail": "t0 to t2", "color": "mint"},
        {"label": "Test 2", "detail": "t2 to t3", "color": "blue"},
        {"label": "Train 3", "detail": "t0 to t3", "color": "mint"},
        {"label": "Test 3", "detail": "t3 to t4", "color": "blue"},
    ],
    title="Walk-Forward Validation"
)
wf.save(f"{BASE}/module_03_time_series/guides/walk_forward_timeline.svg")

print("All 7 SVG diagrams generated successfully.")
