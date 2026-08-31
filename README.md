# GeoKG-Loc

**Structured Spatial-Semantic Reasoning for Natural Language Geographic Localization in Semantically Constrained Urban Environments**

GeoKG-Loc is a structured spatial-semantic reasoning framework that reformulates natural language geographic localization from embedding-based cross-modal retrieval into structured reasoning over a task-oriented urban geographic knowledge graph. It constructs multi-level spatial relations among buildings, roads, and POIs from OpenStreetMap data, employs a large language model to parse reference entities and spatial constraints from natural language descriptions, and estimates user coordinates via candidate entity combinatorial matching and directional-distance back-projection.

## Results

| Metric | Value |
|--------|-------|
| Mean localization error | 13.27 m |
| Median localization error | 8.38 m |
| Top-1 Recall@15m | 71.43% |
| Top-1 Recall@10m | 60.71% |
| Top-1 Recall@5m | 28.57% |

## Architecture

```
Raw Data (OSM buildings, POIs, roads, aerial imagery)
    │
    ▼
┌─────────────────────────────────────────────┐
│ 1. Data Preparation                         │
│   - Road cleaning (clean-road-step1/2.py)  │
│   - Color extraction (color-from-LLM.py)   │
│   - Scene generation (generate_scenes.py)   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ 2. Knowledge Graph Construction             │
│   - Buffer-radius: buildKG.py               │
│   - Delaunay: build_delaunay_kg.py          │
│   Output: Neo4j-importable CSV files        │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ 3. Natural Language Localization            │
│   - LLM parsing (DeepSeek-V3.2)             │
│   - Entity matching against Neo4j KG        │
│   - Coordinate estimation via back-projection│
│   Main scripts:                             │
│     geo_localization_fixed3.py (buffer KG)  │
│     geolocalization_delaunay2.py (Delaunay)  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ 4. Evaluation & Visualization               │
│   - evaluate_localization.py                │
│   - KG visualization (vis/)                  │
│   - Error heatmap & ablation plots          │
└─────────────────────────────────────────────┘
```

## Project Structure

```
GeoKG-Loc/
├── buildKG.py                        # Buffer-radius KG construction
├── build_delaunay_kg.py              # Delaunay-triangulation KG construction
├── geo_localization_fixed3.py        # Localization engine (buffer KG)
├── geolocalization_delaunay2.py      # Localization engine (Delaunay KG)
├── color-from-LLM.py                 # Building color extraction via VLM
├── generate_scenes.py                # Template-based test scene generation
├── generate_scenes_llm.py            # LLM-based test scene generation
├── data_papre/                       # Data preparation scripts
│   ├── clean-road-step1.py           # Road cleaning (WKT parsing, filtering)
│   ├── clean-road-step2.py           # Road cleaning (connectivity merging)
│   └── color-from-LLM.py             # Color extraction (alternative version)
├── KGdata/                           # Pre-built knowledge graphs
│   ├── r50/                          # 50m buffer radius
│   ├── r100/                         # 100m buffer radius (best performance)
│   ├── r300/                         # 300m buffer radius
│   ├── r500/                         # 500m buffer radius
│   ├── r800/                         # 800m buffer radius
│   ├── delaunay/                     # Delaunay triangulation
│   └── vis/                          # Visualization scripts and outputs
│       ├── visualize_r100_kg*.py     # KG structure visualization
│       ├── plot_100m_localization*.py # Localization error heatmap
│       └── plot_buffer_radius_ablation.py # Ablation study plot
├── evaluation/                       # Evaluation scripts and results
│   ├── evaluate_localization.py      # Batch evaluation with metrics
│   ├── evaluate_no_kg_baseline.py    # Ablation: no-KG baseline
│   ├── batch_run.py                  # Batch execution wrapper
│   ├── single_test.py               # Single query test
│   ├── evaluation_results.json       # Pre-computed results
│   └── scenes_100_llm.json           # 100 test scenes with NL descriptions
├── config.example.yaml               # Configuration template
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Requirements

- Python 3.8+
- Neo4j 5.x (for knowledge graph storage)
- An OpenAI-compatible LLM API endpoint (e.g., DeepSeek)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the configuration template:

```bash
cp config.example.yaml config.yaml
```

2. Edit `config.yaml` with your settings:

```yaml
neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "your_password"

llm:
  api_key: "your_api_key"
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"

paths:
  buildings_csv: "buildings_with_colors.csv"
  pois_csv: "pois_raw.csv"
  scenes_json: "evaluation/scenes_100_llm.json"
  output_dir: "./KGdata"
```

> **Warning:** Never commit `config.yaml` or hardcode API keys in source files. The `.gitignore` file excludes `config.yaml` by default.

## Usage

### Step 1: Data Preparation

Prepare building and POI data from OpenStreetMap. Building colors can be extracted from aerial imagery using:

```bash
python color-from-LLM.py
```

Road data can be cleaned using:

```bash
python data_papre/clean-road-step1.py
python data_papre/clean-road-step2.py
```

### Step 2: Knowledge Graph Construction

**Buffer-radius method** (recommended, best performance at r=100m):

```bash
python buildKG.py
```

Configure the buffer radius by modifying `BUILDING_RADII` in the script. Pre-built graphs for r=50, 100, 300, 500, 800m are included in `KGdata/`.

**Delaunay triangulation method**:

```bash
python build_delaunay_kg.py
```

Import the generated CSV files into Neo4j using `neo4j-admin import` or the Neo4j Data Importer.

### Step 3: Generate Test Scenes

Generate 100 test localization points with natural language descriptions:

```bash
python generate_scenes.py          # template-based
python generate_scenes_llm.py     # LLM-enhanced (requires Neo4j running)
```

### Step 4: Localization

Single query test:

```bash
python evaluation/single_test.py
```

Batch evaluation on all 100 test scenes:

```bash
python evaluation/evaluate_localization.py
```

Ablation study (no-KG baseline):

```bash
python evaluation/evaluate_no_kg_baseline.py
```

### Step 5: Visualization

Visualize the knowledge graph structure:

```bash
python KGdata/vis/visualize_r100_kg_with_inset_roads.py
```

Plot localization error heatmap:

```bash
python KGdata/vis/plot_100m_localization_error_heatmap.py
```

Plot buffer radius ablation study:

```bash
python KGdata/vis/plot_buffer_radius_ablation.py
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BUILDING_RADII` | [100] | Buffer radius in meters for Building-Building relations |
| `THRESHOLD_NEAR` | 50 | Distance threshold for Building-POI NEAR relation (m) |
| `num_points` | 100 | Number of test localization points |
| `search_radius_m` | 100 | Search radius for nearby entities (m) |
| `seed` | 42 | Random seed for reproducibility |

## Data Sources

- **OpenStreetMap**: Building footprints, POIs, and road networks (via Overpass API)
- **KITTI-360**: Point cloud data for building color attributes
- Aerial imagery: Satellite tiles for visual color verification

## Citation

If you use this code, please cite:

```bibtex
@article{ma2026geokgloc,
  title={GeoKG-Loc: Structured Spatial-Semantic Reasoning for Natural Language Geographic Localization in Semantically Constrained Urban Environments},
  author={Ma, Kunyang and Zhang, Zheng and Ge, Wen and Zhang, Jinlu and Zhu, Ge and Wang, Lei and Cheng, Yi},
  journal={Knowledge-Based Systems},
  year={2026},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
