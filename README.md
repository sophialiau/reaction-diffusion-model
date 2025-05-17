# Gray-Scott Reaction-Diffusion Model

This project implements the Gray-Scott reaction-diffusion model, which is a mathematical model that describes pattern formation in chemical systems. The model simulates the interaction between two chemical species that diffuse and react with each other.

## Important Notes

This is simply a for leisure. First time programming with Cursor and Agent!

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python gray_scott.py
```

## Parameters

The model has several parameters that can be adjusted to create different patterns:

- `size`: Size of the grid (default: 100x100)
- `dt`: Time step (default: 1.0)
- `Du`: Diffusion rate of chemical U (default: 0.16)
- `Dv`: Diffusion rate of chemical V (default: 0.08)
- `f`: Feed rate (default: 0.035)
- `k`: Kill rate (default: 0.065)

You can modify these parameters in the `GrayScottModel` class initialization to create different patterns.

## Visualization

The simulation shows an animated visualization of the chemical concentration patterns. The color represents the concentration of chemical V, with brighter colors indicating higher concentrations.