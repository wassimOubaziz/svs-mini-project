# Modern Petri Net Simulator

A sophisticated Petri Net simulator built with Python, featuring a modern GUI, parallel processing capabilities, and interactive animations.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Implementation Details](#implementation-details)

## Overview
This application implements a Petri Net simulator with a graphical user interface. It allows users to create, visualize, and analyze Petri nets, including support for cyclic transitions and parallel state space exploration.

## Features
- Modern GUI with Tkinter
- Interactive visualization using Matplotlib
- Real-time token movement animation
- Support for cyclic transitions
- Parallel state space exploration using MPI
- Marking graph generation and visualization
- Transaction history tracking

## Requirements
```
python >= 3.7
mpi4py
tkinter
matplotlib
networkx
numpy
```

## Installation
1. Ensure you have Python installed
2. Install required packages:
```bash
pip install mpi4py matplotlib networkx numpy
```
3. Run the application:
```bash
mpiexec -n 2 python main.py
```

## Code Structure

### Class Definitions

#### Place Class
```python
class Place:
    def __init__(self, name, tokens=0, pos=(0, 0)):
        self.name = name
        self.tokens = tokens
        self.pos = pos
```
Represents a place in the Petri net with:
- `name`: Unique identifier
- `tokens`: Number of tokens in the place
- `pos`: Position coordinates for visualization

#### Transition Class
```python
class Transition:
    def __init__(self, name, inputs, outputs, pos=(0, 0)):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.pos = pos
```
Represents a transition with:
- `inputs`: List of input places
- `outputs`: List of output places
- Methods for checking if enabled and firing transitions

#### MarkingGraph Class
```python
class MarkingGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
```
Manages the marking graph structure for state space exploration.

### Main GUI Class (ModernPetriNetGUI)

#### Initialization
```python
def __init__(self, is_master=True):
    self.is_master = is_master
    # ... GUI setup code
```
Sets up the main application window and initializes GUI components.

#### GUI Components
- Left Panel: Controls for adding places and transitions
- Right Panel: Petri net visualization
- Transaction History: Logs all actions
- Animation Controls: Play/pause/replay functionality

#### Key Methods

##### Animation System
```python
def animate_marking(self, marking_graph):
    # Handles token movement animation
```
- Calculates token movements
- Manages cyclic transitions
- Uses easing functions for smooth animation

##### State Space Exploration
```python
def start_exploration(self):
    # Parallel exploration of reachable states
```
- Uses MPI for parallel processing
- Generates marking graph
- Identifies cycles

##### Visualization
```python
def update_petri_visualization(self):
    # Updates the Petri net display
```
- Draws places, transitions, and tokens
- Handles dynamic updates
- Manages layout

### Animation Details

#### Token Movement
The animation system uses several components:
1. Path calculation between places and transitions
2. Easing functions for smooth movement
3. Cyclic transition handling
4. Frame-by-frame updates

#### Cycle Detection
```python
def has_cycle(edges):
    # Detects cycles in the marking graph
```
- Uses NetworkX for cycle detection
- Ensures proper animation of cyclic paths

## Usage

### Basic Operations
1. Add Places:
   - Enter place name and token count
   - Click "Add Place"

2. Add Transitions:
   - Enter transition name
   - Select input and output places
   - Click "Add Transition"

3. Explore Petri Net:
   - Click "Explore Petri Net" to start analysis
   - View animation of token movements
   - Check transaction history

### Example Usage
```python
# Load example Petri net
def load_example(self):
    # Creates a simple Petri net with cycles
```

## Implementation Details

### Parallel Processing
- Uses MPI for state space exploration
- Master-worker architecture
- Load balancing across processes

### Animation System
- Frame-based animation using Matplotlib
- Smooth token movement with easing
- Proper cycle handling
- Dynamic marking updates

### State Management
- Marking tracking
- Transaction history
- Edge processing for cycles

### Error Handling
- Safe animation cleanup
- Process synchronization
- GUI state management

## Contributing
Feel free to submit issues and enhancement requests.

## License
[MIT License](LICENSE)
