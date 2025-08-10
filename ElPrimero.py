import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime, timedelta
from pytz import timezone
from pysolar.solar import get_altitude, get_azimuth

# Configuración inicial
latitude = -0.2105367  # EPN, Quito
longitude = -78.491614
timezone_local = timezone("America/Guayaquil")

class SolarTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Seguidor Solar 2-DOF - Control Matemático")
        self.root.geometry("1200x800")

        # Frame izquierdo: controles
        control_frame = ttk.LabelFrame(root, text="Configuración", padding=10)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        # Controles de entrada
        ttk.Label(control_frame, text="Fecha:").grid(row=0, column=0, sticky="w", pady=5)
        self.date_entry = DateEntry(control_frame, date_pattern='yyyy-mm-dd')
        self.date_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Hora inicio:").grid(row=1, column=0, sticky="w", pady=5)
        self.hour_spin = ttk.Spinbox(control_frame, from_=0, to=23, width=8)
        self.hour_spin.set(6)
        self.hour_spin.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Duración (h):").grid(row=2, column=0, sticky="w", pady=5)
        self.duration_spin = ttk.Spinbox(control_frame, from_=1, to=12, width=8)
        self.duration_spin.set(12)
        self.duration_spin.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Intervalo (min):").grid(row=3, column=0, sticky="w", pady=5)
        self.interval_spin = ttk.Spinbox(control_frame, from_=1, to=60, width=8)
        self.interval_spin.set(15)
        self.interval_spin.grid(row=3, column=1, padx=5, pady=5)

        # Botón de control
        self.run_button = ttk.Button(control_frame, text="Calcular y Animar", command=self.run_simulation)
        self.run_button.grid(row=4, column=0, columnspan=2, pady=15)

        # Ángulos calculados
        angles_frame = ttk.LabelFrame(control_frame, text="Ángulos Calculados", padding=10)
        angles_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky="we")

        ttk.Label(angles_frame, text="Elevación (θ):").grid(row=0, column=0, sticky="w")
        self.elevation_label = ttk.Label(angles_frame, text="0.00°")
        self.elevation_label.grid(row=0, column=1, sticky="e")

        ttk.Label(angles_frame, text="Azimuth (α):").grid(row=1, column=0, sticky="w")
        self.azimuth_label = ttk.Label(angles_frame, text="0.00°")
        self.azimuth_label.grid(row=1, column=1, sticky="e")

        ttk.Label(angles_frame, text="Pitch:").grid(row=2, column=0, sticky="w")
        self.pitch_label = ttk.Label(angles_frame, text="0.00°")
        self.pitch_label.grid(row=2, column=1, sticky="e")

        ttk.Label(angles_frame, text="Roll:").grid(row=3, column=0, sticky="w")
        self.roll_label = ttk.Label(angles_frame, text="0.00°")
        self.roll_label.grid(row=3, column=1, sticky="e")

        # Frame derecho: visualización
        self.plot_frame = ttk.LabelFrame(root, text="Visualización 3D", padding=10)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Variables de animación
        self.animation = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.sun_vectors = None
        self.times = None
        self.current_frame = 0
        
        # Parámetros del panel
        self.panel_width = 0.5
        self.panel_height = 0.8
        self.sun_distance = 1.0
        self.reference_vector = np.array([0, 1, 0])  # Para orientación consistente

    def clean_previous_animation(self):
        """Limpia la animación anterior"""
        if self.animation:
            self.animation.event_source.stop()
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.fig:
            plt.close(self.fig)
        self.animation = None
        self.canvas = None
        self.fig = None
        self.ax = None

    def calculate_sun_position(self, date, hour_start, duration_hours, time_step_minutes):
        """Calcula la posición solar y ángulos de control"""
        start_time = timezone_local.localize(
            datetime.combine(date, datetime.min.time()) + timedelta(hours=hour_start))
        
        times = []
        elevations = []
        azimuths = []
        pitch_angles = []
        roll_angles = []
        
        for i in range(0, duration_hours * 60, time_step_minutes):
            t = start_time + timedelta(minutes=i)
            elevation = get_altitude(latitude, longitude, t)
            azimuth = get_azimuth(latitude, longitude, t)
            
            # Cálculo de ángulos de control
            pitch = 90 - elevation  # Ángulo de inclinación (0=horizontal, 90=vertical)
            roll = azimuth          # Ángulo de rotación horizontal
            
            times.append(t)
            elevations.append(elevation)
            azimuths.append(azimuth)
            pitch_angles.append(pitch)
            roll_angles.append(roll)
        
        # Convertir a arrays numpy
        elevations = np.array(elevations)
        azimuths = np.array(azimuths)
        
        # Convertir a radianes
        elev_rad = np.radians(elevations)
        azim_rad = np.radians(azimuths)
        
        # Calcular vectores solares unitarios
        sun_vectors = np.array([
            np.cos(elev_rad) * np.sin(azim_rad),
            np.cos(elev_rad) * np.cos(azim_rad),
            np.sin(elev_rad)
        ]).T
        
        return times, sun_vectors, elevations, azimuths, pitch_angles, roll_angles

    def create_panel_vertices(self, normal_vector):
        """Crea los vértices del panel orientado según el vector normal"""
        normal = normal_vector / np.linalg.norm(normal_vector)
        
        # Calcular vectores ortogonales para la orientación del panel
        if np.allclose(np.abs(normal), [0, 0, 1]):
            # Caso especial cuando el panel apunta directamente hacia arriba
            u = np.array([1, 0, 0])
        else:
            u = np.cross(self.reference_vector, normal)
            u = u / np.linalg.norm(u)
        
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        # Calcular vértices del panel
        half_width = self.panel_width / 2
        half_height = self.panel_height / 2
        
        vertices = [
            half_height * u + half_width * v,
            -half_height * u + half_width * v,
            -half_height * u - half_width * v,
            half_height * u - half_width * v
        ]
        
        return [vertices]

    def init_animation(self):
        """Inicializa la animación"""
        # Dibujar trayectoria del sol
        sun_path = self.sun_vectors * self.sun_distance
        self.ax.plot(sun_path[:,0], sun_path[:,1], sun_path[:,2], 
                    'y-', alpha=0.5, marker='o', markersize=3, label="Trayectoria solar")
        
        # Crear panel inicial
        panel_verts = self.create_panel_vertices(self.sun_vectors[0])
        self.panel = Poly3DCollection(panel_verts, color='blue', alpha=0.8, label="Panel solar")
        self.ax.add_collection3d(self.panel)
        
        # Configuración del gráfico
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])
        self.ax.set_zlim([0, 1.5])
        self.ax.set_xlabel("Este-Oeste")
        self.ax.set_ylabel("Norte-Sur")
        self.ax.set_zlabel("Altura")
        self.ax.legend()
        
        return self.panel,

    def update_animation(self, frame):
        """Actualiza el frame de la animación"""
        self.current_frame = frame
        
        # Actualizar panel
        normal = self.sun_vectors[frame]
        panel_verts = self.create_panel_vertices(normal)
        self.panel.set_verts(panel_verts)
        
        # Actualizar rayos solares (limpiar anteriores primero)
        for artist in self.ax.lines[1:]:
            artist.remove()
        for artist in self.ax.collections[1:]:
            artist.remove()
        
        # Dibujar rayo solar
        self.ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], 
                      color='orange', length=1.2, arrow_length_ratio=0.1, label="Rayo solar")
        
        # Dibujar posición del sol
        self.ax.scatter(normal[0]*self.sun_distance, 
                        normal[1]*self.sun_distance, 
                        normal[2]*self.sun_distance,
                        color='yellow', s=200, edgecolor='orange', zorder=10)
        
        # Actualizar ángulos en la interfaz
        time_str = self.times[frame].strftime("%H:%M")
        self.ax.set_title(f"Posición solar a las {time_str}")
        
        self.elevation_label.config(text=f"{self.elevations[frame]:.2f}°")
        self.azimuth_label.config(text=f"{self.azimuths[frame]:.2f}°")
        self.pitch_label.config(text=f"{self.pitch_angles[frame]:.2f}°")
        self.roll_label.config(text=f"{self.roll_angles[frame]:.2f}°")
        
        # Detener al finalizar
        if frame == len(self.times) - 1:
            self.animation.event_source.stop()
        
        return self.panel,

    def run_simulation(self):
        """Ejecuta la simulación completa"""
        try:
            self.clean_previous_animation()
            
            # Obtener parámetros de entrada
            date = self.date_entry.get_date()
            hour_start = int(self.hour_spin.get())
            duration = int(self.duration_spin.get())
            interval = int(self.interval_spin.get())
            
            # Calcular posiciones y ángulos solares
            (self.times, self.sun_vectors, self.elevations, 
             self.azimuths, self.pitch_angles, self.roll_angles) = \
                self.calculate_sun_position(date, hour_start, duration, interval)
            
            # Configurar gráfico 3D
            self.fig = plt.Figure(figsize=(10, 8), dpi=100)
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Configurar canvas
            self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Crear animación
            self.animation = FuncAnimation(
                self.fig, self.update_animation,
                frames=len(self.times),
                init_func=self.init_animation,
                interval=200,
                blit=False,
                repeat=False
            )
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SolarTrackerApp(root)
    root.mainloop()