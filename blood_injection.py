import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider

class BloodInjectionModel:
    def __init__(self, size=100, dt=1.0):
        """
        Initialize the blood injection pattern model.
        
        Parameters:
        - size: Size of the grid (size x size)
        - dt: Time step
        """
        self.size = size
        self.dt = dt
        
        # Initialize the grid
        self.U = np.ones((size, size))  # Activator (injected substance)
        self.V = np.zeros((size, size))  # Inhibitor (tissue resistance)
        self.B = np.zeros((size, size))  # Blood flow
        
        # Set parameters for blood injection
        # Parameters tuned to match tissue diffusion dynamics
        self.Du, self.Dv = 0.16, 0.08  # Diffusion rates
        self.f, self.k = 0.0545, 0.062  # Feed and kill rates
        self.blood_flow_rate = 0.02    # Rate of blood flow
        self.tissue_resistance = 0.01  # Rate of tissue resistance
        
        # Initialize blood injection pattern
        self.init_injection()
            
    def init_injection(self):
        """Initialize blood injection pattern with biologically realistic conditions."""
        # Create initial conditions based on tissue structure
        # Start with a uniform tissue surface
        x = np.linspace(-1, 1, self.size)
        y = np.linspace(-1, 1, self.size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # Create initial injection point
        # This represents the initial injection site
        injection = np.exp(-20 * R)  # Central injection point
        
        # Add initial substance concentration
        self.V = injection
        
        # Initialize tissue resistance
        # Higher in surrounding tissue, lower at injection site
        self.U = 1 - 0.8 * injection
        
        # Initialize blood flow
        # Higher near blood vessels, lower in tissue
        self.B = 0.5 * (1 - np.exp(-5 * R))  # Simulating blood vessel network
        
        # Add biological noise to simulate tissue variability
        noise = np.random.normal(0, 0.02, (self.size, self.size))
        self.V += noise
        self.V = np.clip(self.V, 0, 1)

    def laplacian(self, Z):
        """Calculate the Laplacian of the grid using a 3x3 convolution."""
        Ztop = Z[0:-2, 1:-1]
        Zleft = Z[1:-1, 0:-2]
        Zbottom = Z[2:, 1:-1]
        Zright = Z[1:-1, 2:]
        Zcenter = Z[1:-1, 1:-1]
        return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter)

    def update(self):
        """Update the grid for one time step."""
        # Calculate the Laplacian
        Lu = self.laplacian(self.U)
        Lv = self.laplacian(self.V)
        Lb = self.laplacian(self.B)
        
        # Calculate the reaction terms
        # This represents the interaction between injected substance and tissue
        uv2 = self.U[1:-1, 1:-1] * self.V[1:-1, 1:-1] * self.V[1:-1, 1:-1]
        
        # Update blood flow (diffusion and tissue effects)
        self.B[1:-1, 1:-1] += self.dt * (0.1 * Lb + self.blood_flow_rate * (1 - self.V[1:-1, 1:-1]))
        self.B = np.clip(self.B, 0, 1)
        
        # Tissue resistance effect based on blood flow
        resistance_effect = 1 + self.tissue_resistance * (1 - self.B[1:-1, 1:-1])
        
        # Update the grid
        # Add a small amount of noise to simulate biological variability
        noise = np.random.normal(0, 0.001, (self.size-2, self.size-2))
        
        self.U[1:-1, 1:-1] += self.dt * (self.Du * Lu - uv2 + self.f * (1 - self.U[1:-1, 1:-1])) + noise
        self.V[1:-1, 1:-1] += self.dt * (self.Dv * Lv + uv2 * resistance_effect - (self.f + self.k) * self.V[1:-1, 1:-1]) + noise
        
        # Enforce boundary conditions
        self.U[0, :] = self.U[1, :]
        self.U[-1, :] = self.U[-2, :]
        self.U[:, 0] = self.U[:, 1]
        self.U[:, -1] = self.U[:, -2]
        
        self.V[0, :] = self.V[1, :]
        self.V[-1, :] = self.V[-2, :]
        self.V[:, 0] = self.V[:, 1]
        self.V[:, -1] = self.V[:, -2]
        
        self.B[0, :] = 0  # No blood flow at boundaries
        self.B[-1, :] = 0
        self.B[:, 0] = 0
        self.B[:, -1] = 0

def create_custom_colormap():
    """Create a custom colormap with the specified colors."""
    colors = ['#FFA0AC', '#B4DC7F']  # Pink to Green
    return LinearSegmentedColormap.from_list('custom', colors)

def main():
    # Create the model
    model = BloodInjectionModel(size=100)
    
    # Create figure with white background
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Create custom colormap
    custom_cmap = create_custom_colormap()
    
    # Set up the plot
    img = ax.imshow(model.V, cmap=custom_cmap, interpolation='bilinear')
    plt.colorbar(img, ax=ax)
    ax.set_title('Blood Stream Injection\n(Simulating Tissue Diffusion & Blood Flow)')
    
    # Add speed control slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    speed_slider = Slider(ax_slider, 'Speed', 1, 100, valinit=50)
    
    def update(frame):
        model.update()
        img.set_array(model.V)
        return [img]
    
    # Create the animation with variable speed
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
    
    def update_speed(val):
        anim.event_source.interval = 1000 / val
    
    speed_slider.on_changed(update_speed)
    plt.show()

if __name__ == "__main__":
    main() 