import os

# Lista oficial de los 24 actores de La que se avecina
ACTORES_BASE = [
    "Jordi Sánchez Antonio Recio", "Nathalie Seseña Berta Escobar",
    "Pablo Chiapella Amador Rivas", "Eva Isanta Maite Figueroa",
    "Nacho Guerreros Coque Calatrava", "Macarena Gómez Lola Trujillo",
    "José Luis Gil Enrique Pastor", "Ricardo Arroyo Vicente Maroto",
    "Loles León Menchu Carrascosa", "Petra Martínez Doña Fina",
    "Vanesa Romero Raquel Villanueva", "Fernando Tejero Fermín Trujillo",
    "Miren Ibarguren Yoli Morcillo", "Luis Merlo Bruno Quiroga",
    "Cristina Castaño Judith Becker", "Antonio Pagudo Javier Maroto",
    "Isabel Ordaz Araceli Madariaga", "Malena Alterio Cristina Aguilera",
    "Antonia San Juan Estela Reynolds", "Víctor Palmero Alba Recio",
    "Ernesto Sevilla Teodoro Rivas", "Eduardo Gómez Máximo Angulo",
    "Mariví Bilbao Izaskun Sagastume", "Laura Gómez-Lacueva Greta Garmendia"
]

# Rutas globales
PATHS = {
    "dataset": "dataset",
    "runs": "runs",
    "weights": "weights",
    "models": "models",
    "test": "test",
    "onnx_export": os.path.join("weights", "best.onnx"),
    "labels": os.path.join("WebGPU", "labels.json")
}
