# Use Python base image
FROM python:3.11

# Set working directory inside container
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "user_interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
