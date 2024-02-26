import tkinter as tk
import numpy as np

tkinterinterface = tk.Tk()
tkinterinterface.title("2D Point Transformation")

canvas = tk.Canvas(tkinterinterface, width=300, height=300)
canvas.pack(side=tk.LEFT)

frame_end = tk.Frame(tkinterinterface)
frame_end.pack(side=tk.RIGHT, padx=10)

points = []  

def adding(event):
    x, y = event.x, event.y
    points.append((x, y))
    canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="blue")

canvas.bind("<Button-1>", adding)

def coordinate_system():
    canvas.create_line(0, 150, 300, 150, fill="gray")  # X
    canvas.create_line(150, 0, 150, 300, fill="gray")  # Y

def updating(points):
    canvas.delete("all")
    coordinate_system()
    
    for point in points:
        x, y = point
        
        canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="blue")

coordinate_system()


def perform_transformation():
    try:
        tx = float(translation_x_entry.get())
        ty = float(translation_y_entry.get())
        angle = np.deg2rad(float(rotation_entry.get()))
        scaling = float(scaling_entry.get())
        
        
        points = [(50, 50), (100, 100), (150, 50)]
        
        
        transform_matrix = np.array([
            [scaling * np.cos(angle), -scaling * np.sin(angle), tx],
            [scaling * np.sin(angle), scaling * np.cos(angle), ty],
            [0, 0, 1]
        ])
        
        
        transformed_points = [np.dot(transform_matrix, np.array([x, y, 1]))[:2] for x, y in points]
        
        
        updating(transformed_points)
    except ValueError:
        result_label.config(text="Invalid input values")


translation_x_label = tk.Label(frame_end, text="tx:")
translation_x_label.pack()
translation_x_entry = tk.Entry(frame_end)
translation_x_entry.pack()

translation_y_label = tk.Label(frame_end, text="ty:")
translation_y_label.pack()
translation_y_entry = tk.Entry(frame_end)
translation_y_entry.pack()

rotation_label = tk.Label(frame_end, text="Rotation (degrees):")
rotation_label.pack()
rotation_entry = tk.Entry(frame_end)
rotation_entry.pack()

scaling_label = tk.Label(frame_end, text="Scaling ratio:")
scaling_label.pack()
scaling_entry = tk.Entry(frame_end)
scaling_entry.pack()

transform_button = tk.Button(frame_end, text="Transform", command=perform_transformation)
transform_button.pack()

result_label = tk.Label(frame_end, text="")
result_label.pack()

tkinterinterface.mainloop()
