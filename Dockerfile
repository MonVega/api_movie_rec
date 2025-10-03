# Imagen base con Python 3.13
FROM python:3.13-slim

# Evitar prompts de instalación
ENV DEBIAN_FRONTEND=noninteractive

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y luego instalar (para aprovechar cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar toda la app al contenedor
COPY . .

# Exponer el puerto que espera Hugging Face Spaces
EXPOSE 7860

# Variable de entorno para Hugging Face Spaces
ENV PORT=7860

# Comando de arranque usando uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]