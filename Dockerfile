# Imagen base
FROM python:3.11-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar requisitos
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY pipeline.py .
COPY data.csv .

# Ejecutar el script por defecto
CMD ["python", "pipeline.py"]
