# Usamos una imagen oficial con Python
FROM python:3.10

# Carpeta donde trabajará Docker
WORKDIR /app

# Copiamos los archivos del proyecto al contenedor
COPY . .

# Instalamos dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponemos el puerto 8989
EXPOSE 8989

# Comando para correr la API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8989"]