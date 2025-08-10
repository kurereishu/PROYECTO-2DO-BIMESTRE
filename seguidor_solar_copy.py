import tkinter as tk 
from tkinter import ttk
from tkcalendar import DateEntry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d
from datetime import datetime, timedelta
from pytz import timezone
from pysolar.solar import get_altitude, get_azimuth

# Configuración geográfica
latitude = -0.2105367
longitude = -78.491614
timezone_local = timezone("America/Guayaquil")

class SolarTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Seguidor Solar 2-DOF - Control Matemático")
        self.root.geometry("1200x850")
        
        # === Marco de control ===
        control_frame = ttk.LabelFrame(root, text="Configuración", padding=10)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        
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
        
        self.run_button = ttk.Button(control_frame, text="Calcular y Animar", command=self.run_simulation)
        self.run_button.grid(row=4, column=0, columnspan=2, pady=15)
        
        # === Marco de ángulos ===
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
        
        # === Visualización 3D ===
        self.plot_frame = ttk.LabelFrame(root, text="Visualización 3D", padding=10)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # === Controles de animación ===
        self.controls_frame = ttk.Frame(root)
        self.controls_frame.grid(row=1, column=1, pady=10, sticky="ew")
        
        self.playing = False
        self.play_button = ttk.Button(self.controls_frame, text="▶️", command=self.toggle_play)
        self.play_button.pack(side="left", padx=5)
        
        self.back_button = ttk.Button(self.controls_frame, text="⏪", command=self.step_back)
        self.back_button.pack(side="left", padx=5)
        
        self.forward_button = ttk.Button(self.controls_frame, text="⏩", command=self.step_forward)
        self.forward_button.pack(side="left", padx=5)
        
        self.reset_button = ttk.Button(self.controls_frame, text="🔁 Reiniciar", command=self.reiniciar_animacion)
        self.reset_button.pack(side="left", padx=5)
        
        self.slider = ttk.Scale(self.controls_frame, from_=0, to=0, orient="horizontal", command=self.slider_moved)
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        
        self.time_label = ttk.Label(self.controls_frame, text="--:--")
        self.time_label.pack(side="left", padx=5)
        
        # === Variables ===
        self.animation = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.sun_vectors = None
        self.times = None
        self.current_frame = 0
        self.panel = None
        self.elevations = []
        self.azimuths = []
        self.pitch_angles = []
        self.roll_angles = []
        self.panel_width = 0.5
        self.panel_height = 0.8
        self.sun_distance = 1.0
        self.reference_vector = np.array([0, 1, 0])
        self.angle_text = None
        
        # === Elementos gráficos de ángulos ===
        self.elevation_arc = None
        self.azimuth_arc = None
        self.elevation_angle_label = None
        self.azimuth_angle_label = None
        self.legend_text = None

    def clean_previous_animation(self):
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
        self.panel = None
        self.angle_text = None
        self.elevation_arc = None
        self.azimuth_arc = None
        self.elevation_angle_label = None
        self.azimuth_angle_label = None
        self.legend_text = None

    def calculate_sun_position(self, date, hour_start, duration_hours, time_step_minutes):
        start_time = timezone_local.localize(datetime.combine(date, datetime.min.time()) + timedelta(hours=hour_start))
        times, elevations, azimuths = [], [], []
        pitch_angles, roll_angles = [], []
        for i in range(0, duration_hours * 60, time_step_minutes):
            t = start_time + timedelta(minutes=i)
            elevation = get_altitude(latitude, longitude, t)
            azimuth = get_azimuth(latitude, longitude, t)
            pitch = 90 - elevation
            roll = azimuth
            times.append(t)
            elevations.append(elevation)
            azimuths.append(azimuth)
            pitch_angles.append(pitch)
            roll_angles.append(roll)
        elev_rad = np.radians(elevations)
        azim_rad = np.radians(azimuths)
        sun_vectors = np.array([
            np.cos(elev_rad) * np.sin(azim_rad),
            np.cos(elev_rad) * np.cos(azim_rad),
            np.sin(elev_rad)
        ]).T
        return times, sun_vectors, elevations, azimuths, pitch_angles, roll_angles

    def create_panel_vertices(self, normal_vector):
        normal = normal_vector / np.linalg.norm(normal_vector)
        if np.allclose(np.abs(normal), [0, 0, 1]):
            u = np.array([1, 0, 0])
        else:
            u = np.cross(self.reference_vector, normal)
            u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        hw = self.panel_width / 2
        hh = self.panel_height / 2
        return [[hh * u + hw * v, -hh * u + hw * v, -hh * u - hw * v, hh * u - hw * v]]

    def create_angle_arc(self, start_vec, end_vec, center=np.array([0,0,0]), radius=0.3, num_points=20):
        start_vec = start_vec / np.linalg.norm(start_vec)
        end_vec = end_vec / np.linalg.norm(end_vec)
        normal = np.cross(start_vec, end_vec)
        if np.linalg.norm(normal) == 0:
            normal = np.array([0,0,1])
        else:
            normal = normal / np.linalg.norm(normal)
        u = start_vec
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        angle_start = 0
        angle_end = np.arccos(np.clip(np.dot(start_vec, end_vec), -1.0, 1.0))
        angles = np.linspace(angle_start, angle_end, num_points)
        arc = np.array([center + radius * (np.cos(a) * u + np.sin(a) * v) for a in angles])
        return arc

    def init_animation(self):
        sun_path = self.sun_vectors * self.sun_distance
        self.ax.plot(sun_path[:, 0], sun_path[:, 1], sun_path[:, 2],
                     'y-', alpha=0.5, marker='o', markersize=3, label="Trayectoria solar")
        panel_verts = self.create_panel_vertices(self.sun_vectors[0])
        self.panel = Poly3DCollection(panel_verts, color='green', alpha=0.8) 
        self.ax.add_collection3d(self.panel)
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])
        self.ax.set_zlim([0, 1.5])
        self.ax.set_xlabel("Este-Oeste")
        self.ax.set_ylabel("Norte-Sur")
        self.ax.set_zlabel("Altura")

        # ✅ Leyenda en la gráfica (abajo a la izquierda)
        self.legend_text = self.ax.text2D(
            0.05, 0.05,
            "● = Elevación (rojo)\n● = Azimuth (azul)",
            transform=self.ax.transAxes, fontsize=9, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

        self.ax.legend()
        return self.panel,

    def update_animation(self, frame):
        self.current_frame = frame
        normal = self.sun_vectors[frame]
        elevation = self.elevations[frame]
        azimuth = self.azimuths[frame]

        # Actualizar panel
        self.panel.set_verts(self.create_panel_vertices(normal))

        # === Limpiar arcos anteriores (✅ forma correcta) ===
        if self.elevation_arc is not None:
            self.elevation_arc.remove()
            self.elevation_arc = None
        if self.azimuth_arc is not None:
            self.azimuth_arc.remove()
            self.azimuth_arc = None
        if self.elevation_angle_label is not None:
            self.elevation_angle_label.remove()
            self.elevation_angle_label = None
        if self.azimuth_angle_label is not None:
            self.azimuth_angle_label.remove()
            self.azimuth_angle_label = None

        # Limpiar otros elementos
        for artist in self.ax.lines[1:]:
            artist.remove()
        for artist in self.ax.collections[1:]:
            if artist != self.panel:
                artist.remove()

        # Dibujar vector solar y sol
        self.ax.quiver(0, 0, 0, normal[0], normal[1], normal[2],
                       color='orange', length=1.2, arrow_length_ratio=0.1)
        self.ax.scatter(normal[0] * self.sun_distance, normal[1] * self.sun_distance,
                        normal[2] * self.sun_distance,
                        color='yellow', s=200, edgecolor='orange', zorder=10)

        # === Arco de elevación (rojo) ===
        proj_horizontal = np.array([normal[0], normal[1], 0])
        if np.linalg.norm(proj_horizontal) > 0.01 and elevation > 5:
            arc_elev = self.create_angle_arc(proj_horizontal, normal, radius=0.3, num_points=15)
            self.elevation_arc = art3d.Line3DCollection([arc_elev], colors=['red'], linewidths=2)
            self.ax.add_collection3d(self.elevation_arc)
            mid_point = arc_elev[len(arc_elev)//2]
            self.elevation_angle_label = self.ax.text(
                mid_point[0], mid_point[1], mid_point[2],
                f"{elevation:.1f}°", fontsize=8, color='red', ha='center', va='center'
            )

        # === Arco de azimuth (azul) ===
        north = np.array([0, 1, 0])
        proj_norm = np.array([normal[0], normal[1], 0])
        if np.linalg.norm(proj_norm) > 0.01 and abs(azimuth) > 5:
            arc_azim = self.create_angle_arc(north, proj_norm, radius=0.2, num_points=15)
            self.azimuth_arc = art3d.Line3DCollection([arc_azim], colors=['blue'], linewidths=2)
            self.ax.add_collection3d(self.azimuth_arc)
            mid_point = arc_azim[len(arc_azim)//2]
            self.azimuth_angle_label = self.ax.text(
                mid_point[0], mid_point[1], mid_point[2],
                f"{azimuth:.1f}°", fontsize=8, color='blue', ha='center', va='center'
            )

        # Texto de ángulos (esquina superior)
        if self.angle_text:
            self.angle_text.remove()
        self.angle_text = self.ax.text2D(
            0.05, 0.95,
            f"Elevación: {elevation:.2f}°\n"
            f"Azimuth: {azimuth:.2f}°\n"
            f"Pitch: {self.pitch_angles[frame]:.2f}°\n"
            f"Roll: {self.roll_angles[frame]:.2f}°",
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )

        self.ax.set_title(f"Posición solar a las {self.times[frame].strftime('%H:%M')}")
        self.elevation_label.config(text=f"{elevation:.2f}°")
        self.azimuth_label.config(text=f"{azimuth:.2f}°")
        self.pitch_label.config(text=f"{self.pitch_angles[frame]:.2f}°")
        self.roll_label.config(text=f"{self.roll_angles[frame]:.2f}°")
        self.slider.set(frame)
        self.time_label.config(text=self.times[frame].strftime("%H:%M"))

        self.canvas.draw_idle()  # Más eficiente

        # Detener animación al final
        if frame == len(self.times) - 1:
            if self.animation and self.animation.event_source:
                self.animation.event_source.stop()
            self.playing = False
            self.play_button.config(text="▶️")

        return self.panel,

    def run_simulation(self):
        try:
            self.clean_previous_animation()
            date = self.date_entry.get_date()
            hour_start = int(self.hour_spin.get())
            duration = int(self.duration_spin.get())
            interval = int(self.interval_spin.get())
            (self.times, self.sun_vectors, self.elevations,
             self.azimuths, self.pitch_angles, self.roll_angles) = self.calculate_sun_position(
                date, hour_start, duration, interval)

            self.fig = plt.Figure(figsize=(10, 8), dpi=100)
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.animation = FuncAnimation(
                self.fig, self.update_animation,
                frames=len(self.times),
                init_func=self.init_animation,
                interval=200, blit=False, repeat=False
            )

            self.slider.config(to=len(self.times) - 1)
            self.slider.set(0)
            self.current_frame = 0
            self.playing = False
            self.play_button.config(text="▶️")
            self.time_label.config(text="--:--")
            self.canvas.draw_idle()

        except Exception as e:
            import traceback
            traceback.print_exc()

    def toggle_play(self):
        if not self.animation or not self.animation.event_source:
            return
        if self.playing:
            self.animation.event_source.stop()
            self.play_button.config(text="▶️")
        else:
            # Reiniciar secuencia y avanzar hasta el frame actual
            self.animation.frame_seq = self.animation.new_frame_seq()
            for _ in range(self.current_frame):
                try:
                    next(self.animation.frame_seq)
                except StopIteration:
                    break
            self.animation.event_source.start()
            self.play_button.config(text="⏸️")
        self.playing = not self.playing

    def step_forward(self):
        if self.sun_vectors is None: return
        if self.current_frame < len(self.sun_vectors) - 1:
            self.current_frame += 1
            self.update_animation(self.current_frame)
            # Sincronizar animación
            if self.animation and self.animation.event_source:
                self.animation.frame_seq = self.animation.new_frame_seq()
                for _ in range(self.current_frame):
                    try:
                        next(self.animation.frame_seq)
                    except StopIteration:
                        break
            self.canvas.draw_idle()

    def step_back(self):
        if self.sun_vectors is None: return
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_animation(self.current_frame)
            if self.animation and self.animation.event_source:
                self.animation.frame_seq = self.animation.new_frame_seq()
                for _ in range(self.current_frame):
                    try:
                        next(self.animation.frame_seq)
                    except StopIteration:
                        break
            self.canvas.draw_idle()

    def slider_moved(self, val):
        if self.sun_vectors is None: return
        frame = int(float(val))
        if frame != self.current_frame:
            self.current_frame = frame
            self.update_animation(frame)
            if self.animation and self.animation.event_source:
                self.animation.frame_seq = self.animation.new_frame_seq()
                for _ in range(self.current_frame):
                    try:
                        next(self.animation.frame_seq)
                    except StopIteration:
                        break
            self.canvas.draw_idle()

    def reiniciar_animacion(self):
        if self.sun_vectors is None: return
        self.current_frame = 0
        self.update_animation(0)
        if self.animation and self.animation.event_source:
            self.animation.event_source.stop()
            self.animation.frame_seq = self.animation.new_frame_seq()
        self.playing = False
        self.play_button.config(text="▶️")
        self.slider.set(0)
        self.time_label.config(text=self.times[0].strftime("%H:%M"))

if __name__ == "__main__":
    root = tk.Tk()
    app = SolarTrackerApp(root)
    root.mainloop()