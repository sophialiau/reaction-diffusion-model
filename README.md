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

## Biological Phenomena

The following biological phenomena have been created via respective base parameters.
- zebra_pattern.py: based off of lines + added melanocyte behavior & Turing instability seed points
- leopard_pattern.py: based off of dots + added melanocyte clustering & primary/secondary spot formation
- fingerprint_pattern.py: based off of spiral + added epidermal ridge formation, developmental timing
- bacterial_colony.py: from a central inoculation point shows branching patterns + environmental heterogeneity + added nutrient diffusion & consumption
- cell_membrane.py: simulates cell membrane pattern formation from circular membrane structure, demonstrates lipid bilayer domain formation + added tension dynamics, protein interactions and fluctuations
- melanoma_pattern.py: simulates tumor growth and angiogenesis + tissue invasion dynamics and lesion development
- skin_wrinkling.py: demonstrates skin under mechanical stress accumulation with collagen fiber orientation and tissue resistance. 
- sun_damage.py: imitates sun damage in the skin by simulation UV intensity patterns, melanin response dynamics and DNA damage tracking
- blood_injection.py: visualizes substance injection into tissue, modeling blood flow and tissue resistance from a central injection point. 

All of the above include noise to simulate biological variability.